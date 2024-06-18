import localeimport logging
import time
from pathlib import Path
import contextlib
import os
import pprint as pp
import nltk
import re
import shutil
import time
import torch
from datetime import date, datetime
from os.path import basename, dirname, join
from pathlib import Path

from cleantext import clean
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from libretranslatepy import LibreTranslateAPI
from natsort import natsorted
from spellchecker import SpellChecker
from tqdm.auto import tqdm
import replicate
import pyngrok
import re
import json
from flask import Flask, request, jsonify
from pyngrok import ngrok
from io import BytesIO
from flask_cors import CORS

os.environ["REPLICATE_API_TOKEN"] = "r8_X9BsbBD5oXIVySo9akgM4SFuCZZVATw3ZAlHH"

locale.getpreferredencoding = lambda: "UTF-8"

import getpass

from pyngrok import ngrok, conf

print("Enter your authtoken, which can be copied from https://dashboard.ngrok.com/auth")
conf.get_default().auth_token = getpass.getpass()

# Open a TCP ngrok tunnel to the SSH server
connection_string = ngrok.connect(22, "tcp").public_url

ssh_url, port = connection_string.strip("tcp://").split(":")
print(f" * ngrok tunnel available, access with `ssh root@{ssh_url} -p{port}`")

import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S",
)



def simple_rename(filepath, target_ext=".txt"):
    _fp = Path(filepath)
    basename = _fp.stem
    return f"OCR_{basename}_{target_ext}"


def rm_local_text_files(name_contains="RESULT_"):
    """
    rm_local_text_files - remove local text files

    Args:
        name_contains (str, optional): [description]. Defaults to "OCR_".
    """
    files = [
        f
        for f in Path.cwd().iterdir()
        if f.is_file() and f.suffix == ".txt" and name_contains in f.name
    ]
    logging.info(f"removing {len(files)} text files")
    for f in files:
        os.remove(f)
    logging.info("done")


def corr(
    s: str,
    add_space_when_numerics=False,
    exceptions=["e.g.", "i.e.", "etc.", "cf.", "vs.", "p."],
) -> str:
    """corrects spacing in a string

    Args:
        s (str): the string to correct
        add_space_when_numerics (bool, optional): [add a space when a period is between two numbers, example 5.73]. Defaults to False.
        exceptions (list, optional): [do not change these substrings]. Defaults to ['e.g.', 'i.e.', 'etc.', 'cf.', 'vs.', 'p.'].

    Returns:
        str: the corrected string
    """
    if add_space_when_numerics:
        s = re.sub(r"(\d)\.(\d)", r"\1. \2", s)

    s = re.sub(r"\s+", " ", s)
    s = re.sub(r'\s([?.!"](?:\s|$))', r"\1", s)

    # fix space before apostrophe
    s = re.sub(r"\s\'", r"'", s)
    # fix space after apostrophe
    s = re.sub(r"'\s", r"'", s)
    # fix space before comma
    s = re.sub(r"\s,", r",", s)

    for e in exceptions:
        expected_sub = re.sub(r"\s", "", e)
        s = s.replace(expected_sub, e)

    return s


def fix_punct_spaces(string):
    """
    fix_punct_spaces - replace spaces around punctuation with punctuation. For example, "hello , there" -> "hello, there"

    Parameters
    ----------
    string : str, required, input string to be corrected

    Returns
    -------
    str, corrected string
    """

    fix_spaces = re.compile(r"\s*([?!.,]+(?:\s+[?!.,]+)*)\s*")
    string = fix_spaces.sub(lambda x: "{} ".format(x.group(1).replace(" ", "")), string)
    string = string.replace(" ' ", "'")
    string = string.replace(' " ', '"')
    return string.strip()


def clean_OCR(ugly_text: str):
    """
    clean_OCR - clean the OCR text files.

    Parameters
    ----------
    ugly_text : str, required, input string to be cleaned

    Returns
    -------
    str, cleaned string
    """
    # Remove all the newlines.
    cleaned_text = ugly_text.replace("\n", " ")
    # Remove all the tabs.
    cleaned_text = cleaned_text.replace("\t", " ")
    # Remove all the double spaces.
    cleaned_text = cleaned_text.replace("  ", " ")
    # Remove all the spaces at the beginning of the text.
    cleaned_text = cleaned_text.lstrip()
    # remove all instances of "- " and " - "
    cleaned_text = cleaned_text.replace("- ", "")
    cleaned_text = cleaned_text.replace(" -", "")
    return fix_punct_spaces(cleaned_text)


def move2completed(from_dir, filename, new_folder="completed", verbose=False):

    # this is the better version
    old_filepath = join(from_dir, filename)

    new_filedirectory = join(from_dir, new_folder)

    if not os.path.isdir(new_filedirectory):
        os.mkdir(new_filedirectory)
        if verbose:
            print("created new directory for files at: \n", new_filedirectory)
    new_filepath = join(new_filedirectory, filename)

    try:
        shutil.move(old_filepath, new_filepath)
        logging.info("successfully moved the file {} to */completed.".format(filename))
    except:
        logging.info(
            "ERROR! unable to move file to \n{}. Please investigate".format(
                new_filepath
            )
        )


"""## pdf2text functions

"""


custom_replace_list = {
    "t0": "to",
    "'$": "'s",
    ",,": ", ",
    "_ ": " ",
    " '": "'",
}

replace_corr_exceptions = {
    "i. e.": "i.e.",
    "e. g.": "e.g.",
    "e. g": "e.g.",
    " ,": ",",
}


spell = SpellChecker()


def check_word_spelling(word: str) -> bool:
    """
    check_word_spelling - check the spelling of a word

    Args:
        word (str): word to check

    Returns:
        bool: True if word is spelled correctly, False if not
    """

    misspelled = spell.unknown([word])

    return len(misspelled) == 0


def eval_and_replace(text: str, match_token: str = "- ") -> str:
    """
    eval_and_replace  - conditionally replace all instances of a substring in a string based on whether the eliminated substring results in a valid word

    Args:
        text (str): text to evaluate
        match_token (str, optional): token to replace. Defaults to "- ".

    Returns:
        str:  text with replaced tokens
    """

    try:
        if match_token not in text:
            return text
        else:
            while True:
                full_before_text = text.split(match_token, maxsplit=1)[0]
                before_text = [
                    char for char in full_before_text.split()[-1] if char.isalpha()
                ]
                before_text = "".join(before_text)
                full_after_text = text.split(match_token, maxsplit=1)[-1]
                after_text = [char for char in full_after_text.split()[0] if char.isalpha()]
                after_text = "".join(after_text)
                full_text = before_text + after_text
                if check_word_spelling(full_text):
                    text = full_before_text + full_after_text
                else:
                    text = full_before_text + " " + full_after_text
                if match_token not in text:
                    break
    except Exception as e:
        logging.error(f"Error spell-checking OCR output, returning default text:\t{e}")
    return text


def cleantxt_ocr(ugly_text, lower=False, lang: str = "en") -> str:
    """
    cleantxt_ocr - clean text from OCR

    Args:
        ugly_text (str): text to clean
        lower (bool, optional): _description_. Defaults to False.
        lang (str, optional): _description_. Defaults to "en".

    Returns:
        str: cleaned text
    """
    # a wrapper for clean text with options different than default

    # https://pypi.org/project/clean-text/
    cleaned_text = clean(
        ugly_text,
        fix_unicode=True,  # fix various unicode errors
        to_ascii=True,  # transliterate to closest ASCII representation
        lower=lower,  # lowercase text
        no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
        no_urls=True,  # replace all URLs with a special token
        no_emails=True,  # replace all email addresses with a special token
        no_phone_numbers=False,  # replace all phone numbers with a special token
        no_numbers=False,  # replace all numbers with a special token
        no_digits=False,  # replace all digits with a special token
        no_currency_symbols=False,  # replace all currency symbols with a special token
        no_punct=False,  # remove punctuations
        replace_with_punct="",  # instead of removing punctuations you may replace them
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        replace_with_number="<NUM>",
        replace_with_digit="0",
        replace_with_currency_symbol="<CUR>",
        lang=lang,  # set to 'de' for German special handling
    )

    return cleaned_text


def format_ocr_out(OCR_data):

    if isinstance(OCR_data, list):
        text = " ".join(OCR_data)
    else:
        text = str(OCR_data)
    _clean = cleantxt_ocr(text)
    return corr(_clean)


def postprocess(text: str) -> str:
    """to be used after recombining the lines"""

    proc = corr(cleantxt_ocr(text))

    for k, v in custom_replace_list.items():
        proc = proc.replace(str(k), str(v))

    proc = corr(proc)

    for k, v in replace_corr_exceptions.items():
        proc = proc.replace(str(k), str(v))

    return eval_and_replace(proc)


def result2text(result, as_text=False) -> str or list:
    """Convert OCR result to text"""

    full_doc = []
    for i, page in enumerate(result.pages, start=1):
        text = ""
        for block in page.blocks:
            text += "\n\t"
            for line in block.lines:
                for word in line.words:
                    # print(dir(word))
                    text += word.value + " "
        full_doc.append(text)

    return "\n".join(full_doc) if as_text else full_doc


def convert_PDF_to_Text(
    PDF_file,
    ocr_model=None,
    max_pages: int = 20,
):

    st = time.perf_counter()
    PDF_file = Path(PDF_file)
    ocr_model = ocr_predictor(pretrained=True) if ocr_model is None else ocr_model
    logging.info(f"starting OCR on {PDF_file.name}")
    doc = DocumentFile.from_pdf(PDF_file)
    truncated = False
    if len(doc) > max_pages:
        logging.warning(
            f"PDF has {len(doc)} pages, which is more than {max_pages}.. truncating"
        )
        doc = doc[:max_pages]
        truncated = True

    # Analyze
    logging.info(f"running OCR on {len(doc)} pages")
    result = ocr_model(doc)
    raw_text = result2text(result)
    proc_text = [format_ocr_out(r) for r in raw_text]
    fin_text = [postprocess(t) for t in proc_text]

    ocr_results = "\n\n".join(fin_text)

    fn_rt = time.perf_counter() - st

    logging.info("OCR complete")

    results_dict = {
        "num_pages": len(doc),
        "runtime": round(fn_rt, 2),
        "date": str(date.today()),
        "converted_text": ocr_results,
        "truncated": truncated,
        "length": len(ocr_results),
    }

    return results_dict


# @title translation functions

lt = LibreTranslateAPI("https://translate.astian.org/")


def translate_text(text, source_l, target_l="en"):

    return str(lt.translate(text, source_l, target_l))


def translate_doc(filepath, lang_start, lang_end="en", verbose=False):
    """translate a document from lang_start to lang_end

        {'code': 'en', 'name': 'English'},
    {'code': 'fr', 'name': 'French'},
    {'code': 'de', 'name': 'German'},
    {'code': 'it', 'name': 'Italian'},"""

    src_folder = dirname(filepath)
    src_folder = Path(src_folder)
    trgt_folder = src_folder / f"translated_{lang_end}"
    trgt_folder.mkdir(exist_ok=True)
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        foreign_t = f.readlines()
    in_name = basename(filepath)
    translated_doc = []
    for line in tqdm(
        foreign_t, total=len(foreign_t), desc="translating {}...".format(in_name[:10])
    ):
        translated_line = translate_text(line, lang_start, lang_end)
        translated_doc.append(translated_line)
    t_out_name = "[To {}]".format(lang_end) + simple_rename(in_name) + ".txt"
    out_path = join(trgt_folder, t_out_name)
    with open(out_path, "w", encoding="utf-8", errors="ignore") as f_o:
        f_o.writelines(translated_doc)
    if verbose:
        print("finished translating the document! - ", datetime.now())
    return out_path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

_here = Path().parent

def load_uploaded_file(file_obj, temp_dir: Path = None):
    # check if mysterious file object is a list
    if isinstance(file_obj, list):
        file_obj = file_obj[0]
    file_path = Path(file_obj.name)

    if temp_dir is None:
        _temp_dir = _here / "temp"
    _temp_dir.mkdir(exist_ok=True)

    try:
        pdf_bytes_obj = open(file_path, "rb").read()
        temp_path = temp_dir / file_path.name if temp_dir else file_path
        # save to PDF file
        with open(temp_path, "wb") as f:
            f.write(pdf_bytes_obj)
        logging.info(f"Saved uploaded file to {temp_path}")
        return str(temp_path.resolve())

    except Exception as e:
        logging.error(f"Trying to load file with path {file_path}, error: {e}")
        print(f"Trying to load file with path {file_path}, error: {e}")
        return None


def convert_PDF(
    pdf_obj,
    language: str = "en",
    max_pages=20,
):
    """
    convert_PDF - convert a PDF file to text

    Args:
        pdf_bytes_obj (bytes): PDF file contents
        language (str, optional): Language to use for OCR. Defaults to "en".

    Returns:
        str, the PDF file contents as text
    """
    # clear local text cache
    rm_local_text_files()
    global ocr_model
    st = time.perf_counter()
    if isinstance(pdf_obj, list):
        pdf_obj = pdf_obj[0]
    file_path = Path(pdf_obj.name)
    if not file_path.suffix == ".pdf":
        logging.error(f"File {file_path} is not a PDF file")

        html_error = f"""
        <div style="color: red; font-size: 20px; font-weight: bold;">
        File {file_path} is not a PDF file. Please upload a PDF file.
        </div>
        """
        return "File is not a PDF file", html_error, None

    conversion_stats = convert_PDF_to_Text(
        file_path,
        ocr_model=ocr_model,
        max_pages=max_pages,
    )
    converted_txt = conversion_stats["converted_text"]
    num_pages = conversion_stats["num_pages"]
    was_truncated = conversion_stats["truncated"]
    # if alt_lang: # TODO: fix this

    rt = round((time.perf_counter() - st) / 60, 2)
    print(f"Runtime: {rt} minutes")
    html = ""
    if was_truncated:
        html += f"<p>WARNING - PDF was truncated to {max_pages} pages</p>"
    html += f"<p>Runtime: {rt} minutes on CPU for {num_pages} pages</p>"

    _output_name = f"RESULT_{file_path.stem}_OCR.txt"
    with open(_output_name, "w", encoding="utf-8", errors="ignore") as f:
        f.write(converted_txt)

    return converted_txt, html, _output_name



logging.info("Starting app")

use_GPU = torch.cuda.is_available()
logging.info(f"Using GPU status: {use_GPU}")
logging.info("Loading OCR model")

ocr_model = ocr_predictor(
    "db_resnet50",
    "crnn_mobilenet_v3_large",
    pretrained=True,
    assume_straight_pages=True,
)

def process_ocr_text(ocr_text):
    questions = []

    current_question = {}
    for line in ocr_text.split("\n"):
        print(line)
        if re.match(r'^\d+\. ', line):
            if current_question:
                questions.append(current_question)  # Append the previous question before creating a new one
            current_question = {"q": line[2:].strip()}  # Create a new question dictionary
        elif line.startswith("A)"):
            current_question["o1"] = line[3:].strip()
        elif line.startswith("B)"):
            current_question["o2"] = line[3:].strip()
        elif line.startswith("C)"):
            current_question["o3"] = line[3:].strip()
        elif line.startswith("D)"):
            current_question["o4"] = line[3:].strip()

    # Append the last question after the loop ends
    if current_question:
        questions.append(current_question)

    return json.dumps(questions, indent=4)


app = Flask(__name__)

port = 8070

public_url = ngrok.connect(port).public_url
print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))

app.config["BASE_URL"] = public_url
CORS(app)

@app.route('/OCR', methods=['POST'])
def get_OCR():
    try:
        uploaded_file = request.files['pdf']

        print(uploaded_file)

        if uploaded_file.mimetype == 'application/pdf':
            pdf_data = uploaded_file.read()

            filename = "temp.pdf"
            with open(filename, 'wb') as f:
                f.write(pdf_data)

        # define pdf bytes as None
        pdf_obj = _here / "temp.pdf"
        pdf_obj = str(pdf_obj.resolve())
        pdf_obj = open(pdf_obj, "rb")
        _temp_dir = _here / "temp"
        _temp_dir.mkdir(exist_ok=True)

        OCR_text, out_placeholder, text_file = convert_PDF(pdf_obj)

        print(OCR_text)


        # Your base prompt and OCR_text
        base_prompt = """make sense out of this ocr data. Just format it into a question paper like questions and options in the format Q, A, B, C, D
            Like:
                1. A boat..
                A) Option B
                B) Option A
                C) Option C
                B) Option D

                2. A car..
                A)..
                ...

                and so on.

          its supposed to be a mcq test. use latex entirely and make sure its correct, only add questions and options ignore the visit my page and other marketing stuff don't assume stuff , no need to answer these questions just format them properly"""



        # Define a variable to store the processed OCR text
        processed_text =  ""

        # Example replicate event loop
        for event in replicate.stream(
            "mistralai/mixtral-8x7b-instruct-v0.1",
            input={
                "top_k": 50,
                "top_p": 0.9,
                "prompt": base_prompt + OCR_text,
                "temperature": 0.8,
                "max_new_tokens": 2048,
                "prompt_template": "<s>[INST] {prompt} [/INST] ",
                "presence_penalty": 0,
                "frequency_penalty": 0
            },
        ):
            # Store the processed OCR text
            processed_text += str(event)



        json_data = process_ocr_text(processed_text)
        return jsonify(json_data)

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port="8070")