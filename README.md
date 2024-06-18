# Test Management System

## Video Tutorial

For a detailed walkthrough of the features and functionality of the Test Management System, please refer to our video tutorial: [Watch Video Tutorial](https://drive.google.com/file/d/1uDU3SHUYOXYfbbCX-S6X6rTmMajZzcvS/view?usp=sharing)

## Services

### 1. Main Backend Server

The main backend server handles user authentication, user profiles, test creation, test analytics, and test uploading. It serves as the core backend infrastructure for the entire system.

#### Installation Process:

1. Navigate to the main-backend-server directory.
2. Install dependencies using npm install.
3. Start the server using nodemon.

### 2. OCR Server

The OCR (Optical Character Recognition) server is a Flask-based service responsible for processing scanned PDF files. It utilizes an LLM (Language Model) called Mistral 7b to extract and interpret text from the scanned documents. The OCR server outputs data in a specific format, including equations rendered using LaTeX.

#### Installation Process:

1. Navigate to the ocr-server directory.
2. Create a virtual environment using python -m venv venv.
3. Activate the virtual environment:
   - On Windows: venv\Scripts\activate
   - On macOS and Linux: source venv/bin/activate
4. Install dependencies using pip install -r requirements.txt.
5. Start the server using python app.py.

### 3. Main Test Frontend

The main test frontend provides a user interface for creating tests, previewing them, and managing test analytics. Users can create tests with various question types, including text and image-based questions. They can also analyze the performance of test-takers through the provided analytics.

#### Installation Process:

1. Navigate to the main-test-frontend directory.
2. Install dependencies using npm install.
3. Start the server using npm run start.

### 4. Test Portal Frontend

The test portal frontend serves as a secure environment for test-takers to attempt exams. It opens when a user starts a test on the main frontend and maintains full-screen mode while preventing access to developer tools. The UI of the test portal mimics the layout of standardized tests like the JEE examination. Test-takers' responses are saved in Redis for the duration of the test to ensure persistence.

#### Installation Process:

1. Navigate to the test-portal-frontend directory.
2. Install dependencies using npm install.
3. Start the server using npm start.

## Requirements

To run the codebase, you'll need the following:

- Node.js
- Python 3.x
- Redis (for the Test Portal Frontend)

## Roadmap

- [ ] Complete Analytics Route: Implement additional features and enhancements to the test analytics functionality.
- [ ] Enhance OCR Quality: Improve the OCR server's ability to process scanned PDFs and extract text accurately.
- [ ] Implement Additional Security Measures: Enhance security measures to ensure the integrity and confidentiality of test data.
