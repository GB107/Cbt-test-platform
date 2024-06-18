import React, { useState } from 'react';
import Radio from '@mui/material/Radio';
import RadioGroup from '@mui/material/RadioGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import TextField from '@mui/material/TextField';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Button from '@mui/material/Button';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import AddCircleIcon from '@mui/icons-material/AddCircle';
import DeleteIcon from '@mui/icons-material/Delete';
import axios from 'axios';
import Dialog from '@mui/material/Dialog';
import ListItemText from '@mui/material/ListItemText';
import ListItemButton from '@mui/material/ListItemButton';
import List from '@mui/material/List';
import Divider from '@mui/material/Divider';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';
import CloseIcon from '@mui/icons-material/Close';
import Slide from '@mui/material/Slide';
import { TransitionProps } from '@mui/material/transitions';
import Textarea from '@mui/joy/Textarea';
import { MathJax, MathJaxContext } from 'better-react-mathjax';
import Latex from 'react-latex-next';
import LinearProgress from '@mui/material/LinearProgress';

function LinearIndeterminate() {
  return (
    <Box sx={{ width: '100%' }}>
      <LinearProgress />
    </Box>
  );
}

enum InputMethod {
  MANUAL = 'manual',
  PDF = 'pdf',
}

interface Question {
  question: string;
  options: string[];
  image: File | null; // Image file
  isImage: boolean; // Indicates whether an image has been added
  correctAnswer: string; // Correct answer for the question
}

interface Section {
  name: string;
  marks: number;
  positiveMarks: number;
  negativeMarks: number;
  schema: string;
  questionType: string;
  questions: Question[];
}

const Transition = React.forwardRef(function Transition(
  props: TransitionProps & { children: React.ReactElement },
  ref: React.Ref<unknown>,
) {
  return <Slide direction="up" ref={ref} {...props} />;
});

const TestPage: React.FC = () => {
  const [selectedInputMethod, setSelectedInputMethod] = useState<InputMethod>(
    InputMethod.PDF
  );
  const [sections, setSections] = useState<Section[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [currentSection, setCurrentSection] = useState<number>(0);
  const [manualInstructions, setManualInstructions] = useState<string>('');
  const [pdfInstructions, setPdfInstructions] = useState<string>('');
  const [loader, setLoader] = useState<boolean>(false);
  const [testName, setTestName] = useState<string>('');
  const [testDuration, setTestDuration] = useState<string>('');

  const [open, setOpen] = useState(false);

  const handleClickOpen = () => {
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
  };

  const handleAddSection = () => {
    setSections([
      ...sections,
      {
        name: `Section ${sections.length + 1}`,
        marks: 0,
        positiveMarks: 0,
        negativeMarks: 0,
        schema: '',
        questionType: '',
        questions: [],
      },
    ]);
  };

  const handleInputChange = (
    sectionIndex: number,
    questionIndex: number,
    value: string
  ) => {
    const newSections = [...sections];
    newSections[sectionIndex].questions[questionIndex].question = value;
    setSections(newSections);
  };

  const handleOptionChange = (
    sectionIndex: number,
    questionIndex: number,
    optionIndex: number,
    value: string
  ) => {
    const newSections = [...sections];
    newSections[sectionIndex].questions[questionIndex].options[
      optionIndex
    ] = value;
    setSections(newSections);
  };

  const handleImageChange = (
    sectionIndex: number,
    questionIndex: number,
    image: File | null
  ) => {
    const newSections = [...sections];
    newSections[sectionIndex].questions[questionIndex].image = image;
    newSections[sectionIndex].questions[questionIndex].isImage = !!image;
    setSections(newSections);
  };

  const handleCorrectAnswerChange = (
    sectionIndex: number,
    questionIndex: number,
    value: string
  ) => {
    const newSections = [...sections];
    newSections[sectionIndex].questions[questionIndex].correctAnswer = value;
    setSections(newSections);
  };

  const handleManualSubmit = () => {
    console.log('Manual Instructions:', manualInstructions);
    console.log('Manual Answers:', sections);
    
    try {
      axios.post('http://localhost:5000/add-exam', {
        testName: testName,
        testDuration: testDuration,
        manualInstructions: manualInstructions, 
        sections: sections
      }).then(response => {
        console.log('Test added successfully:', response.data);
      }).catch(error => {
        console.error('Error adding test:', error);
      });
    } catch (error) {
      console.error('Error adding test:', error);
    }
  };
  

  const handlePDFSubmit = async () => {
    if (!selectedFile) {
      console.error('No file selected');
      return;
    }

    const formData = new FormData();
    formData.append('pdf', selectedFile);

    try {
      setLoader(true);
      
      const response = await axios.post('https://23c1-34-86-160-132.ngrok-free.app/OCR', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });


      console.log('OCR Processed:', response.data);

      const manualdata = JSON.parse(response.data)

      setLoader(false);


      const sectionData = {
        name: `Section 1`,  
        marks: 0,
        positiveMarks: 0,
        negativeMarks: 0,
        schema: '',
        questionType: '',
        questions: manualdata.map((data) => ({
          question: data.q,
          options: [data.o1, data.o2, data.o3, data.o4],
          image: null,
          isImage: false,
          correctAnswer: '',
        })),
      };

      setSections([sectionData]);

      setSelectedInputMethod(InputMethod.MANUAL);

    } catch (error) {
      console.error('Error processing OCR:', error);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleAddQuestion = (sectionIndex: number) => {
    const newSections = [...sections];
    newSections[sectionIndex].questions.push({
      question: '',
      options: ['', '', '', ''],
      image: null,
      isImage: false,
      correctAnswer: '', // Initialize correct answer as empty string
    });
    setSections(newSections);
  };

  const handleDeleteQuestion = (sectionIndex: number, questionIndex: number) => {
    const newSections = [...sections];
    newSections[sectionIndex].questions.splice(questionIndex, 1);
    setSections(newSections);
  };

  const handlePreview = () => {
    console.log('Previewing...');
    // Add preview logic here
  };

  const boxWidth = `${sections.length * 200 + 1000}px`;

  return (
    <Box
      width={boxWidth}
      mx="auto"
      p={4}
      border="0.5px solid #ccc"
      borderRadius={4}
      bgcolor="#f7f7f7"
      color="#000"
      mt={4}
    >
      <div style={{ display: 'flex', justifyContent: 'center' }}>
        <h1>Create Test</h1>
      </div>
      <Box mb={2}>
        <RadioGroup
          aria-label="inputMethod"
          defaultValue={InputMethod.PDF}
          value={selectedInputMethod}
          onChange={(e) =>
            setSelectedInputMethod(e.target.value as InputMethod)
          }
          row
        >
          <FormControlLabel
            value={InputMethod.MANUAL}
            control={<Radio />}
            label="Upload"
          />
          <FormControlLabel
            value={InputMethod.PDF}
            control={<Radio />}
            label="Upload PDF"
          />
        </RadioGroup>
      </Box>
      {selectedInputMethod === InputMethod.MANUAL && (
        <>
          <Textarea
            placeholder="Add instructions for manual input here..."
            value={manualInstructions}
            onChange={(e) => setManualInstructions(e.target.value)}
            rows={4}
            style={{ width: '100%', marginBottom: '10px' }}
          />
          <Box>
            <TextField
              label="Test Name"
              variant="outlined"
              value={testName}
              onChange={(e) => setTestName(e.target.value)}
              style={{ marginRight: '10px' }}
            />
            <TextField
              label="Test Duration"
              variant="outlined"
              value={testDuration}
              onChange={(e) => setTestDuration(e.target.value)}
            />
          </Box>
          <Tabs
            value={currentSection}
            onChange={(e, newValue) => setCurrentSection(newValue)}
            textColor="primary"
          >
            {sections.map((section, index) => (
              <Tab key={index} label={section.name} />
            ))}
            <Button
              variant="contained"
              onClick={handleAddSection}
              style={{
                marginLeft: 'auto',
                backgroundColor: '#4caf50',
                color: '#fff',
              }}
            >
              Add Section
            </Button>
          </Tabs>
          {sections.map((section, sectionIndex) => (
            <Box key={sectionIndex} hidden={sectionIndex !== currentSection}>
              <Box maxHeight="400px" overflow="auto">
                <h2>{section.name}</h2>
                <Grid container spacing={2} alignItems="center">
                  <Grid item xs={4}>
                    <TextField
                      label="Section Name"
                      variant="outlined"
                      value={section.name}
                      onChange={(e) => {
                        const newSections = [...sections];
                        newSections[sectionIndex].name = e.target.value;
                        setSections(newSections);
                      }}
                    />
                  </Grid>
                  <Grid item xs={4}>
                    <TextField
                      label="Positive Marks"
                      variant="outlined"
                      value={section.positiveMarks}
                      onChange={(e) => {
                        const newSections = [...sections];
                        const positiveMarks = parseInt(e.target.value);
                        newSections[sectionIndex].positiveMarks = isNaN(positiveMarks)
                          ? 0
                          : positiveMarks;
                        setSections(newSections);
                      }}
                    />
                  </Grid>
                  <Grid item xs={4}>
                    <TextField
                      label="Negative Marks"
                      variant="outlined"
                      value={section.negativeMarks}
                      onChange={(e) => {
                        const newSections = [...sections];
                        const negativeMarks = parseInt(e.target.value);
                        newSections[sectionIndex].negativeMarks = isNaN(negativeMarks)
                          ? 0
                          : negativeMarks;
                        setSections(newSections);
                      }}
                    />
                  </Grid>
                </Grid>
                {section.questions.map((question, questionIndex) => (
                  <Box key={questionIndex} mt={2}>
                    <div>
                      <strong>Question {questionIndex + 1}:</strong>
                      <Textarea
                        placeholder={`Question ${questionIndex + 1
                          }...`}
                        style={{ width: '100%', minHeight: '100px' }}
                        value={question.question}
                        onChange={(e) =>
                          handleInputChange(
                            sectionIndex,
                            questionIndex,
                            e.target.value
                          )
                        }
                      />
                    </div>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={(e) =>
                        handleImageChange(
                          sectionIndex,
                          questionIndex,
                          e.target.files && e.target.files.length > 0
                            ? e.target.files[0]
                            : null
                        )
                      }
                      style={{ marginTop: '10px' }}
                    />
                    <br />
                    <br />
                    <Grid container spacing={2}>
                      {[...Array(4).keys()].map((optionIndex) => (
                        <Grid item xs={3} key={optionIndex}>
                          <TextField
                            id={`option-${sectionIndex}-${questionIndex}-${optionIndex}`}
                            label={`Option ${optionIndex + 1}`}
                            variant="outlined"
                            value={question.options[optionIndex]}
                            onChange={(e) =>
                              handleOptionChange(
                                sectionIndex,
                                questionIndex,
                                optionIndex,
                                e.target.value
                              )
                            }
                          />
                        </Grid>
                      ))}
                    </Grid>
                    <TextField
                      label="Correct Answer"
                      variant="outlined"
                      value={question.correctAnswer}
                      onChange={(e) =>
                        handleCorrectAnswerChange(
                          sectionIndex,
                          questionIndex,
                          e.target.value
                        )
                      }
                      style={{ marginTop: '10px' }}
                    />
                    <br />
                    <Button
                      variant="contained"
                      color="error"
                      onClick={() =>
                        handleDeleteQuestion(
                          sectionIndex,
                          questionIndex
                        )
                      }
                      style={{ marginTop: '10px' }}
                      startIcon={<DeleteIcon />}
                    >
                      Delete
                    </Button>
                  </Box>
                ))}
                <br />
                <Button
                  variant="contained"
                  onClick={() =>
                    handleAddQuestion(sectionIndex)
                  }
                  style={{
                    backgroundColor: '#4caf50',
                    color: '#fff',
                  }}
                  startIcon={<AddCircleIcon />}
                >
                  Add Question
                </Button>
              </Box>
            </Box>
          ))}

              <Box mt={2} sx={{
                display: 'flex',
                justifyContent: "flex-end"
              }}
    
              >
            <Button
              variant="contained"
              onClick={handleManualSubmit}
              style={{
                marginRight: '10px',
                backgroundColor: '#2196f3',
                color: '#fff',
              }}
            >
              Submit
            </Button>
            <Button
              variant="contained"
              onClick={handleClickOpen}
              style={{
                marginLeft: '10px',
                backgroundColor: '#ff9800',
                color: '#fff',
              }}
            >
              Preview
            </Button>
            </Box>
            <Dialog
              fullScreen
              open={open}
              onClose={handleClose}
              TransitionComponent={Transition}
            >
              <AppBar sx={{ position: 'relative' }}>
                <Toolbar>
                  <IconButton
                    edge="start"
                    color="inherit"
                    onClick={handleClose}
                    aria-label="close"
                  >
                    <CloseIcon />
                  </IconButton>
                  <Typography sx={{ ml: 2, flex: 1 }} variant="h6" component="div">
                    Preview
                  </Typography>
                  {/* Add a cancel button to close the dialog */}
                  <Button autoFocus color="inherit" onClick={handleClose}>
                    Cancel
                  </Button>
                </Toolbar>
              </AppBar>
              <List
                sx={{
                  p: 5
                }}
              >
                {/* Render the test data stored in the sections state */}
                {sections.map((section, sectionIndex) => (
                  <div key={sectionIndex}

                  >
                    <Typography variant="h6">{section.name}</Typography>
                    {section.questions.map((question, questionIndex) => (
                      <div key={questionIndex}
                        style={{
                          marginBottom: '40px'
                        }}
                      >
                        <Typography variant="subtitle1">
                            <Latex>
                              {`Question ${questionIndex + 1}: ${question.question}`}
                            </Latex>
                        </Typography>
                        <ul>
                          {question.options.map((option, optionIndex) => (
                            <li key={optionIndex}>
                              <MathJaxContext>
                                <MathJax inline dynamic>
                                  {`Option ${optionIndex + 1}: ${option}`}
                                </MathJax>
                              </MathJaxContext></li>
                          ))}
                        </ul>
                      </div>
                    ))}
                  </div>
                ))}
              </List>
            </Dialog>

        </>
      )}
      {selectedInputMethod === InputMethod.PDF && (
        <>
          <Textarea
            placeholder="Add instructions for PDF upload here..."
            value={pdfInstructions}
            onChange={(e) => setPdfInstructions(e.target.value)}
            rows={4}
            style={{ width: '100%', marginBottom: '10px' }}
          />
          <div>
            <h2>Upload PDF</h2>
                        <Box
                          sx={{
                            display: "flex",
                            justifyContent: "space-between"
                          }}
                        >
            {loader && <LinearIndeterminate />}
            {!loader && (
              <>
              <input
                type="file"
                accept="application/pdf"
                onChange={handleFileChange}
              />
            <div>
              <Button
                variant="contained"
                onClick={handlePDFSubmit}
                disabled={!selectedFile}
                style={{
                  backgroundColor: '#2196f3',
                  color: '#fff',
                }}
              >
                Submit
              </Button>
              <Button
                variant="contained"
                onClick={handlePreview}
                style={{
                  marginLeft: '10px',
                  backgroundColor: '#ff9800',
                  color: '#fff',
                }}
              >
                Preview
              </Button>
              </div>
              </>
              )}
              </Box>
            </div>

        </>
      )}
    </Box>
  );
};

export default TestPage;
