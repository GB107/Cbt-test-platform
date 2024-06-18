import React, { useEffect, useState } from 'react';
import "./App.css"
import { MathJaxContext, MathJax } from "better-react-mathjax";
import axios from 'axios';
import Modal from '@mui/material/Modal';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';

interface Question {
  q: string;
  o1: string;
  o2: string;
  o3: string;
  o4: string;
  ans: string;
  f?: boolean;
}

const style = {
  position: 'absolute' as 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  width: 400,
  bgcolor: 'background.paper',
  border: '2px solid #000',
  boxShadow: 24,
  p: 4,
};


interface TopicData {
  [key: string]: [Question[]];
}

interface ApiData {
    [key: string]: [Question[]];
}

// const data = {
//     "Arihant Complex Numbers": [
//         {
//             "ans": "",
//             "o1": "a",
//             "o2": "a",
//             "o3": "a",
//             "o4": "a",
//             "q": "Express in the form \\( A+i B \\) :\n(i) \\( \\frac{(a+i)^{2}}{(a-i)}-\\frac{(a-i)^{2}}{(a+i)} \\)\n(ii) \\( \\left(\\frac{1+\\sin \\alpha+i \\cos \\alpha}{1+\\sin \\alpha-i \\cos \\alpha}\\right)^{n} \\)\n(iii) \\( \\left(\\frac{1+2 i}{1+i}\\right)^{n} \\) for \\( n=\\pm 1, \\pm 2, \\pm 3, \\pm 4, \\)\n(iv) \\( \\log _{e}(1+i \\tan \\alpha) \\)\n"
//         },
//         {
//             "ans": "",
//             "o1": "a",
//             "o2": "a",
//             "o3": "a",
//             "o4": "a",
//             "q": "Express in the modulus-amplitude form :\n(i) \\( 1-i \\)\n(ii) \\( \\frac{(\\sqrt{3}-1)+i(\\sqrt{3}+1)}{2 \\sqrt{2}} \\)\n(iii) \\( \\frac{(1+i)^{2 n+1}}{(1-i)^{2 n-1}}, n \\in N \\)\n(iv) \\( 1+i \\tan \\theta(-\\pi<\\theta<\\pi, \\theta \\neq \\pm \\pi / 2) \\)\n"
//         },
//         {
//             "ans": "",
//             "o1": "a",
//             "o2": "a",
//             "o3": "a",
//             "o4": "a",
//             "q": "Find the square root of :\n(i) \\( -i \\)\n(ii) \\( -7-24 i \\)\n(iii) \\( x+\\sqrt{\\left(-x^{4}-x^{2}-1\\right)} \\)\n(iv) \\( \\left(\\frac{a}{b}\\right)^{2}+\\left(\\frac{b}{a}\\right)^{2}-\\frac{4}{i}\\left(\\frac{a}{b}-\\frac{b}{a}\\right)-6 \\)"
//         },
//         {
//             "ans": "",
//             "o1": "a",
//             "o2": "a",
//             "o3": "a",
//             "o4": "a",
//             "q": "Find the integral solution of the equations:\n(i) \\( \\quad(1-i)^{n}=2^{n} \\)\n(ii) \\( (1+i)^{n}=(1-i)^{n} \\)\n"
//         },
//         {
//             "ans": "",
//             "o1": "real\n",
//             "o2": "purely imaginary\n(ii) Find real \\( \\theta \\) such that \\( \\frac{1+i \\cos \\theta}{1-2 i \\cos \\theta} \\) is purely imaginary.\n",
//             "o3": "a",
//             "o4": "a",
//             "q": "(i) Find real \\( \\theta \\) such that \\( \\frac{3+2 i \\sin \\theta}{1-2 i \\sin \\theta} \\) is\n"
//         },
//         {
//             "ans": "",
//             "o1": "a",
//             "o2": "a",
//             "o3": "a",
//             "o4": "a",
//             "q": "(i) If \\( x=-2-\\sqrt{-3}, \\) find the value of \\( 2 x^{4}+5 x^{3}+7 x^{2}-x+41 \\)\n(ii) If \\( x=\\frac{3+5 i}{2}, \\) find the value of \\( 2 x^{3}+2 x^{2}-7 x+72 . \\) Show that it will \\( b \\)\n\\[\n\\text { unaltered if } x=\\frac{3-5 i}{2}\n\\]\n"
//         },
//         {
//             "ans": "",
//             "o1": "a",
//             "o2": "a",
//             "o3": "a",
//             "o4": "a",
//             "q": "If \\( 1, \\omega, \\omega^{2} \\) are the three cube roots of unity, show that\n(i) \\( \\left(1-\\omega+\\omega^{2}\\right)\\left(1-\\omega^{2}+\\omega^{4}\\right)\\left(1-\\omega^{4}+\\omega^{8}\\right) \\ldots \\) to \\( 2 n \\) factors \\( =2^{2 n} \\)\n(ii) \\( \\left(a+b \\omega+c \\omega^{2}\\right)^{3}+\\left(a+b \\omega^{2}+c \\omega\\right)^{3} \\)\n\\[\n\\begin{array}{l}\n{=(2 a-b-c)(2 b-c-a)(2 c-a-b)} \\\\\n{=27 a b c \\text { if } a+b+c=0}\n\\end{array}\n\\]"
//         },
//         {
//             "ans": "",
//             "o1": "a",
//             "o2": "a",
//             "o3": "a",
//             "o4": "a",
//             "q": "If \\( i z^{3}+z^{2}-z+i=0, \\) then show that \\( |z|=1 \\)\n"
//         },
//         {
//             "ans": "",
//             "o1": "a",
//             "o2": "a",
//             "o3": "a",
//             "o4": "a",
//             "q": "Prove that the complex numbers \\( z_{1}, z_{2} \\) and the origin form an equilateral triangle if and only if \\( \\frac{z_{1}}{z_{2}}+\\frac{z_{2}}{z_{1}}=1 \\)\n"
//         },
//         {
//             "ans": "",
//             "o1": "a",
//             "o2": "a",
//             "o3": "a",
//             "o4": "a",
//             "q": "If \\( |z|=1, \\) then prove that\n\\[\n\\arg \\left(z^{2}+z\\right)=\\frac{1}{2} \\arg (z)\n\\]\n"
//         },
//         {
//             "ans": "",
//             "o1": "a",
//             "o2": "a",
//             "o3": "a",
//             "o4": "a",
//             "q": "If \\( \\alpha, \\beta \\) are non-real complex cube roots of unity, then prove that\n\\[\n\\alpha^{4}+\\beta^{4}+\\alpha^{-1} \\beta^{-1}=0\n\\]\n"
//         },
//         {
//             "ans": "",
//             "o1": "a",
//             "o2": "a",
//             "o3": "a",
//             "o4": "a",
//             "q": "If \\( \\left|z_{1}\\right|=\\left|z_{2}\\right|=\\ldots=\\left|z_{n}\\right|=1, \\) prove that\n\\[\n\\left|z_{1}+z_{2}+\\ldots+z_{n}\\right|=\\left|\\frac{1}{z_{1}}+\\frac{1}{z_{2}}+\\ldots+\\frac{1}{z_{n}}\\right|\n\\]"
//         },
//         {
//             "ans": "",
//             "o1": "a",
//             "o2": "a",
//             "o3": "a",
//             "o4": "a",
//             "q": "If \\( \\alpha, \\beta \\) are the roots of the equation \\( x^{2}-2 x+4=0, \\) prove that\n\\[\n\\alpha^{n}+\\beta^{n}=2^{n+1} \\cos \\left(\\frac{n \\pi}{3}\\right)\n\\]\n"
//         },
//         {
//             "ans": "",
//             "o1": "a",
//             "o2": "a",
//             "o3": "a",
//             "o4": "a",
//             "q": "Construct an equation whose roots are \\( n \\) th power of the roots of the equation\n\\[\nx^{2}-2 x \\cos \\theta+1=0\n\\]\n"
//         },
//         {
//             "ans": "",
//             "o1": "a",
//             "o2": "a",
//             "o3": "a",
//             "o4": "a",
//             "q": "If \\( x=e^{i \\theta} \\) and \\( \\sqrt{\\left(1-c^{2}\\right)}=n c-1 \\) then prove that\n\\[\n1+c \\cos \\theta=\\frac{c}{2 n}(1+n x)\\left(1+\\frac{n}{x}\\right)\n\\]\n"
//         },
//         {
//             "ans": "",
//             "o1": "a",
//             "o2": "a",
//             "o3": "a",
//             "o4": "a",
//             "q": "Show that a real value of \\( x \\) will satisfy the equation\n\\[\n\\frac{1-i x}{1+i x}=a-i b \\text { if } a^{2}+b^{2}=1\n\\]\n"
//         },
//         {
//             "ans": "",
//             "o1": "a",
//             "o2": "a",
//             "o3": "a",
//             "o4": "a",
//             "q": "Prove that\n\\[\n\\left|1-\\overline{z_{1}} z_{2}\\right|^{2}-\\left|z_{1}-z_{2}\\right|^{2}=\\left(1-\\left|z_{1}\\right|^{2}\\right)\\left(1-\\left|z_{2}\\right|^{2}\\right)\n\\]\n"
//         },     
//     ],
//     "Complex Numbers": [
//         {
//             "ans": "",
//             "o1": "a",
//             "o2": "a",
//             "o3": "a",
//             "o4": "a",
//             "q": "If \\( z \\) is a complex number and \\( \\alpha=\\frac{1+z}{1+\\bar{z}} \\) then prove that \\( \\alpha^{n}+\\frac{1}{\\alpha^{n}}=2 \\cos n(\\text { amp } z) \\)\n"
//         },
//         {
//             "ans": "",
//             "o1": "a",
//             "o2": "a",
//             "o3": "a",
//             "o4": "a",
//             "q": "The equation \\( z^{2}+a z+b=0 \\) has a purely imaginary root where \\( a \\) and \\( b \\) are complex constants. Prove that\n\\[\n(a+\\bar{a})(a \\bar{b}+\\bar{a} b)+(b-\\bar{b})^{2}=0\n\\]\n"
//         },
//         {
//             "ans": "",
//             "o1": "a",
//             "o2": "a",
//             "o3": "a",
//             "o4": "a",
//             "q": "Show that the roots of the equation \\( (1+x)^{2 n}+(1-x)^{2 n}=0 \\) are given by\n\\[\n\\begin{array}{l}\n{\\quad x=\\pm i \\tan \\{(2 k-1) \\pi / 2 n\\}} \\\\\n{\\text { where } k=1,2,3, \\ldots, n}\n\\end{array}\n\\]"
//         },
//         {
//             "ans": "",
//             "o1": "a",
//             "o2": "a",
//             "o3": "a",
//             "o4": "a",
//             "q": "Locate the complex number \\( z=x+i y \\) for which\n(i) \\( z^{2}+\\bar{z}^{2}+2|z|^{2}<8 i(\\bar{z}-z) \\)\n(ii) \\( \\log _{\\sqrt{3}}"
//             }
//     ]
// }




const App: React.FC = () => {

  const [apiData, setApiData] = useState<ApiData>({});
  const [topicData, setTopicData] = useState<TopicData>({});
  const [topics, setTopics] = useState<string[]>([]);
  const [currentTopic, setCurrentTopic] = useState<string>('');
  const [currentSlide, setCurrentSlide] = useState<number>(0);
  const [examStatus, setExamStatus] = useState<Object>({});
  const [option, setOption] = useState<string>('');
  const [startTime, setStartTime] = useState<number>(0);
  const [data, setData] = useState({});
  const [examName, setExamName] = useState<string>('');
  const [Instructions, setInstructions] = useState<string>('');
  const [testId,setTestId] = useState<string>('');
  const [userId,setUserId] = useState<string>('');
  
  function convertData(originalData) {
    const convertedData = {};

    // Iterate through sections
    originalData.sections.forEach((section, index) => {
        const sectionName = section.name;
        const sectionData = [];

        // Iterate through questions in the section
        section.questions.forEach(question => {
            const questionData = {
                ans: question.correctAnswer || "", // Assuming correct answer is present
                o1: question.options[0] || "", // Assuming at least one option is present
                o2: question.options[1] || "",
                o3: question.options[2] || "",
                o4: question.options[3] || "",
                q: question.question || ""
            };

            sectionData.push(questionData);
        });

        convertedData[sectionName] = sectionData;
    });

    return convertedData;
}



useEffect(() => {
  const getUrlParameter = (name) => {
    const urlParams = new URLSearchParams(window.location.search);
    console.log(urlParams);
    return urlParams.get(name);
  };

  // Retrieve testId and userId from URL
  const testId = getUrlParameter('testId');
  const userId = getUrlParameter('userId');
  setTestId(testId);
  setUserId(userId);
  

  // Do whatever you need with testId and userId
  console.log('testId:', testId);
  console.log('userId:', userId);

  const fetchData = async () => {
    try {
      const response = await axios.post('http://localhost:5000/qp', {
        testId: testId,
        userId: userId,
      });
      setExamName(response.data.testname);
      setInstructions(response.data.instructions);

      console.log(response.data, "response.data");

      const apidata = convertData(response.data);
      console.log(apidata, "apidata");
      
      setApiData(apidata);
      setData(apidata);

      const topicNames = Object.keys(apidata); // Use apidata here
      console.log(topicNames);
      setTopics(topicNames);
      setCurrentTopic(topicNames[0]);

  
      //make a object of exam status that looks like apiData and its inner object looks like {status: '', timeSpent: '', answer: ''} with the index
      setExamStatus(() => {
          const examStatus = {};
          topicNames.forEach((topic) => {
              examStatus[topic] = apidata[topic].map(() => ({ status: 'nv', timeSpent: '', answer: '' }));
          });
          return examStatus;
          }
      );

    } catch (error) {
      console.error(error);
    }
  };

  fetchData();

  // Cleanup function to abort axios request if component unmounts
  return () => {
    // Cleanup code if needed
  };

}, []);




  useEffect(() => {
    setTopicData(apiData[currentTopic] || []);
  }, [apiData, currentTopic]);
  

  const loadTopic = (topic: string) => {
    setCurrentTopic(topic);
    setCurrentSlide(0);
    const updatedExamStatus = { ...examStatus };
    updatedExamStatus[topic][0].status = 'na';
    setExamStatus(updatedExamStatus);
  };




  const showSlide = (n: number) => {
    console.log(examStatus);
    //here
    setOption('');
    const updatedExamStatus = { ...examStatus };
    if (updatedExamStatus[currentTopic][currentSlide].answer === '') {
      updatedExamStatus[currentTopic][currentSlide].status = 'na';
    }
    setExamStatus(updatedExamStatus);
    //
    // console.log(examStatus, "examStatus");
    try {
      const data = axios.post('http://localhost:5000/sub', {
        examStatus: examStatus,
        testId: testId,
        userId: userId,
      });
      console.log('Submission successful:');
    } catch (error) {
      console.error('Submission failed:', error);
    }
    

    if (n >= topicData.length) {
      // If slide exceeds topicData length, it should not navigate to next topic
      return;
    }
    setCurrentSlide(n);
    console.log(examStatus);
  };




  const showNextSlide = () => {
    showSlide(currentSlide + 1);
  };






const clearResponse = () => {
  const updatedExamStatus = { ...examStatus };
  setOption('');
  updatedExamStatus[currentTopic][currentSlide].answer = '';
  updatedExamStatus[currentTopic][currentSlide].status = 'na';
  setExamStatus(updatedExamStatus);
};


const markAndNextSlide = () => {
  // Mark the current slide for review and navigate to the next slide
  const updatedExamStatus = { ...examStatus };
  updatedExamStatus[currentTopic][currentSlide].status = 'mr';
  if (option !== '') {
    updatedExamStatus[currentTopic][currentSlide].answer = option;
    updatedExamStatus[currentTopic][currentSlide].status = 'amr';
  }
  setExamStatus(updatedExamStatus);
  showNextSlide();
};


const saveAndNextSlide = () => {
    // Save the current response and navigate to the next slide
    const updatedExamStatus = { ...examStatus };
    updatedExamStatus[currentTopic][currentSlide].answer = option;
    updatedExamStatus[currentTopic][currentSlide].status = 'an';
    if (option === '') {
      updatedExamStatus[currentTopic][currentSlide].status = 'na';
    }
    setExamStatus(updatedExamStatus);
    showNextSlide();
}


  const showPreviousSlide = () => {
    if (currentSlide === 0) {
      // If current slide is 0, it should not go to previous topic
      return;
    }
    showSlide(currentSlide - 1);
  };

  const showResults = () => {
    //modal here
    // Implement showing results logic
    //on modal same api
  };

  const [modal, setModal] = useState<boolean>(true);
  const handleClose = () => setModal(false);




return (
    <div>
            <Modal
        open={modal}
        // onClose={handleClose}
        aria-labelledby="modal-modal-title"
        aria-describedby="modal-modal-description"
      >
        <Box sx={style}>
          <Typography id="modal-modal-title" variant="h6" component="h2"
            fontFamily={'Open Sans'}
          >
            Instructions
          </Typography>
          <Typography id="modal-modal-description" sx={{ mt: 2 }}
          fontFamily={'Open Sans'}
          >
           {Instructions}
          </Typography>

          <Box
            sx={{ display: 'flex', justifyContent: 'flex-end' }}
          >
          <Button
            sx={{ mt: 2 }}
            variant="contained"
            color='success'
          onClick={handleClose}>Agree and Start Test</Button>
          </Box>
        </Box>
      </Modal>
    <header style={{ backgroundColor: '#3b5998'}}>
        <h1
            style={{
                color: 'white',
                fontSize: '32px',
                fontWeight: 'bold',
                fontFamily: 'Arial, sans-serif',
                margin: '0',
                padding: '10px',
            }}
        >{examName}</h1>
    </header>
    <div id="header_1">
        Paper1
        <div className="just_2">
            <div>Instructions</div>
        </div>
        <div className="just_2">
            <div>Question Paper</div>
        </div>
    </div>
    <div id="exam_name" className="left">
        <div className="exam_names">JEE Advanced 2019</div>
    </div>
    <div id="time" className="left">
        Section
        <div id="time_left">Time Left:
            zstate
        </div>
    </div>
    <div id="view_in_lang" className="left">
        <select id="lang_select" className="just_3">
            <option value="english">English</option>
            <option value="hindi">Hindi</option>
        </select>
        <div className="just_3">View in:</div>
    </div>

        <div id="section_names" className="left">
            {topics.map((topic: string) => (
                <div
                    key={topic}
                    className={currentTopic === topic ? 'section_selected' : 'section_unselected'}
            onClick={() => loadTopic(topic)}
          >
            {topic}
          </div>
        ))}
      </div>

      {/* Render question area */}
      <div id="question_area_scrollable" className="left">
        <div className="question-title">
          <div id="question-title">Question no. {currentSlide + 1}</div>
        </div>
        <div id="area_with_background">
          <img
            id="section_info_img"
            src="https://www.digialm.com/per/g01/pub/379/ASM/OnlineAssessment/M2/tkcimages/EP1S1.jpg"
            alt="Section Info"
          />
          <div id="quiz">
            {Object.entries(topicData).map(([key, value], index) => (
              <div key={key} className={`slide ${index === currentSlide ? 'active-slide' : ''}`}>
                <div className="question">
                    <MathJaxContext>
                        <MathJax inline dynamic>
                            {value.q}
                        </MathJax>
                    </MathJaxContext>

                </div>
                <div className="answers">
                  <label>
                    <input type="radio" key={key + index} value="A" onClick={() => setOption('A')} checked={option === 'A'
                      || examStatus[currentTopic][currentSlide].answer === 'A'
                  } />
                    {" " + value.o1}
                  </label>
                  <label>
                    <input type="radio" key={key + index} value="B" onClick={() => setOption('B')} checked={option === 'B' ||
                      examStatus[currentTopic][currentSlide].answer === 'B'
                  } />
                    {" " + value.o2}
                  </label>
                  <label>
                    <input type="radio" key={key + index} value="C" onClick={() => setOption('C')} checked={option === 'C' ||
                      examStatus[currentTopic][currentSlide].answer === 'C'
                  } />
                    {" " + value.o3}
                  </label>
                  <label>
                    <input type="radio" key={key + index} value="D" onClick={() => setOption('D')} checked={option === 'D' || 
                      examStatus[currentTopic][currentSlide].answer === 'D'
                  } />
                    {" " + value.o4}
                  </label>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div id="next_options" className="left">
        <div id="mfran" className="button" onClick={markAndNextSlide}>
          Mark for Review & Next
        </div>
        <div id="cr" className="button" onClick={clearResponse}>
          Clear Response
        </div>
        <div
          id="pre"
          className="button"
          onClick={showPreviousSlide}
        >
          Previous
        </div>
        <div id="next" className="button" onClick={saveAndNextSlide}>
          Save and Next
        </div>
      </div>


        <div id="sidebar">

        <div id="person_info">
            <div><img id="profile_img" src="https://www.digialm.com//OnlineAssessment/images/NewCandidateImage.jpg" /></div>
            <div id="cname">John Smith</div>
        </div>
        <div id="border">
            <div id="color_info">
                <div className="just_4">
                    <span className="just_5" style={{ backgroundPosition: '-7px -55px' }}>
                        {
                            Object.entries(examStatus).reduce((acc, [topic, questions]) => {
                                return acc + questions.filter((question) => question.status === 'an' || question.status === 'amr').length;
                            }, 0)
                        }
                    </span>
                    <span className="just_6">Answered</span>
                </div>
                <div className="just_4">
                    <span className="just_5" style={{ backgroundPosition: '-42px -56px' }}>   
                    {                    
                     Object.entries(examStatus).reduce((acc, [topic, questions]) => {
                                                return acc + questions.filter((question) => question.status === 'na').length;
                                            }, 0)
                    }
                    </span>
                    <span className="just_6">Not Answered</span>
                </div>
                <div className="just_4">
                    <span className="just_5" style={{ backgroundPosition: '-107px -56px' }}>
                    {
                        Object.entries(examStatus).reduce((acc, [topic, questions]) => {
                            return acc + questions.filter((question) => question.status === 'nv').length;
                        }, 0)
                    }
                    </span>
                    <span className="just_6">Not Visited</span>
                </div>
                <div className="just_4">
                    <span className="just_5" style={{ backgroundPosition: '-75px -54px' }}>{
                        //all topics
                        Object.entries(examStatus).reduce((acc, [topic, questions]) => {
                            return acc + questions.filter((question) => question.status === 'mr').length;
                        }, 0)
                    }
                    </span>
                    <span className="just_6">Marked for Review</span>
                </div>
                <div className="just_4" id="long">
                    <span className="just_5" style={{ backgroundPosition: '-9px -87px' }}>{
                        Object.entries(examStatus).reduce((acc, [topic, questions]) => {
                            return acc + questions.filter((question) => question.status === 'amr').length;
                        }
                        , 0)
}
                    </span>
                    <span className="just_6">Answered & Marked for Review (will be considered for evaluation)</span>
                </div>
            </div>
            <div id="small_header">{currentTopic}</div>
            <div id="questions_select_area">
                <div id = "choose_text">Choose a Question</div>
                <div id="palette-list">
                    {
                        Object.entries(examStatus[currentTopic] || {}).map(([key, value], index) => (
                            <div
                                key={key}
                                className={`item ${value.status}`}
                                onClick={() => {showSlide(index)}}
                            >
                                {index + 1}
                            </div>
                        ))
                    }
                </div>
            </div>	
        </div>
        </div>

      {/* Render submit container */}
      <div id="submit_container">
        <div className="button" id="submit" onClick={showResults}>
          Submit
        </div>
      </div>


      {/* Render footer */}
      <footer id="ultimate_footer"></footer>
    </div>
  );
};

export default App;
