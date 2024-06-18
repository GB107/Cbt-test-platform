// const ExamModel = require('../Models/exam.js');

const Test = require('../Models/exam.js');
const Users = require('../Models/users.js');
const UserTest = require('../Models/examstatus.js'); // Import the UserTest model
// const examstatus = require('../Models/examstatus.js');


class examController{


            


    static saveAnswer = async (req, res) => {
        try {
            const { testId, userId, examStatus } = req.body;

            let userTest = await UserTest.findOne({ userId, testId });
    
            if (!userTest) {
                userTest = new UserTest({
                    userId,
                    testId,
                    examStatus,
                });
            } else {
                userTest.examStatus = examStatus; 
            }
            await userTest.save();
            res.status(200).json({ message: 'Answers saved successfully' });
        } catch (error) {
            console.error('Error saving answers:', error);
            res.status(500).json({ error: 'Failed to save answers' });
        }
    };
    
    static getStatusCounts = async (req, res) => {
        try {
            const { userId } = req.body;
            const userTests = await UserTest.find({ userId });
    
            const statusCountsResponse = [];
    
            for (const userTest of userTests) {
                const { testId, examStatus } = userTest;
                const test = await Test.findOne({ _id: testId }); 
    
                if (test) {
                    const testName = test.testname;
                    const counts = { na: 0, nv: 0, amr: 0, mr: 0, an: 0 };
    
                    examStatus.forEach(section => {
                        Object.values(section).forEach(questions => {
                            questions.forEach(question => {
                                counts[question.status]++; 
                            });
                        });
                    });
                    statusCountsResponse.push({ testName, testId, counts });
                }
            }
            console.log(statusCountsResponse);
            res.status(200).json(statusCountsResponse);
        } catch (error) {
            console.error('Error retrieving status counts:', error);
            res.status(500).json({ error: 'Failed to retrieve status counts' });
        }
    };
    

    // const Test = require('./testModel'); // Import the Test model

    static addExam = async (req, res) => {
        try {
            const { manualInstructions, sections,testName,testDuration } = req.body;
            console.log(manualInstructions)
            // console.log(sections)
            console.log(testDuration)
            const test = new Test({
                instructions: manualInstructions,
                testname: testName,
                //covert testduration to a number
                testduration: testDuration ? parseInt(testDuration) : 0,
                sections: sections,
                createdby: "Admin",
                ratings: 0,
                difficulty: "Medium"
            });
    
            const savedTest = await test.save();
            res.status(201).json({
                isSuccessful: true,
                data: savedTest,
            });
        } catch (error) {
            console.error('Error creating test:', error);
            res.status(500).json({
                isSuccessful: false,
                data: error.message || 'Internal server error',
            });
        }
    };
    
    
    static getExams = (req, res) => {
        const { userProfile } = req.body;
        const userId = userProfile.userId;
    
        console.log(userId, "user");
    
        Users.findOne({ id: userId })

            .then((user) => {
                if (!user) {
                    return res.status(404).json({ message: "User not found" });
                }
    
                Test.find().then((tests) => {
                    // Map over all the tests and return the test_id, testname, ratings, and difficulty

                    
                    const testList = tests.map(test => {             
                        
                        //section wise get postive marks and negative marks and number of questions

                        let sctionsWiseData = [];

                        test.sections.forEach(section => {
                            sctionsWiseData.push({
                                sectionName: section.name,
                                positiveMarks: section.positiveMarks,
                                negativeMarks: section.negativeMarks,
                                totalQuestions: section.questions.length
                            });
                        })

                        return {
                            test_id: test._id,
                            testname: test.testname,
                            ratings: test.ratings,
                            difficulty: test.difficulty,
                            totalQuestions: test.sections.reduce((total, section) => total + section.questions.length, 0),
                            sectionsWiseData: sctionsWiseData
                        };
                    });
    
                    res.status(200).json({ tests: testList });
                }).catch((err) => {
                    console.log(err);
                    res.status(500).json({ message: "Internal server error" });
                });
            }).catch((err) => {
                console.log(err);
                res.status(500).json({ message: "Internal server error" });
            });
    };
    
    
    static getOneExam = async (req, res) => {
        const{testId, userId} = req.body;
        // const { id } = req.params; // Assuming you get the exam ID from request params
        try {
            const users = await Users.findOne({ id: userId });
            if (!users) {
                return res.status(404).json({ message: "User not found" });
            }

        Test.findOne({ _id: testId })
        .then((exam) => {
            if (!exam) {
                return res.status(404).json({ message: "Exam not found" });
            }
            // users.test.test_id.push(testId);
            users.save();
                res.status(200).json({
                    sections: exam.sections,
                    instructions: exam.instructions,
                    testduration: exam.testduration,
                    testname: exam.testname,
                    // You can include other fields as needed
                });
            })
            .catch((err) => {
                console.log(err);
                res.status(500).json({ message: "Internal server error" });
            });

        } catch (err) {
            console.log(err);
            res.status(500).json({ message: "Internal server error" });
        }
    };

 
}

module.exports = examController;