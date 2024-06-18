import { useEffect, useState } from 'react';
import { Outlet, Link, useNavigate } from "react-router-dom";


import { login, logout, makeApiCallWithUserToken, starterKitIsConfiguredCorrectly, authressLoginClient } from '../authressClient';
import reactLogo from './assets/react.svg';
import authressLogo from './assets/logo.svg';
import Openapi from './openapi';

import React from 'react';
import Button from '@mui/material/Button';
import TestPage from '../components/testportal';
import { Box } from '@mui/material';
import ResponsiveAppBar from '../components/appbar';
import CollapsibleTable from '../components/browsetests';
import BasicLineChart from '../components/analytics';
import axios from 'axios';

function App() {
  const navigate = useNavigate();
  const [userProfile, setUserProfile] = useState({});
  const [authressApiUrlIsSet, setAuthressApiUrlIsSet] = useState(true);
  
  const [showTestPage, setShowTestPage] = useState<boolean>(true);
  const [showAnalyticsPage, setShowAnalyticsPage] = useState<boolean>(false);
  const [showCreatePage, setShowCreatePage] = useState<boolean>(false);
  const [tests, setTests] = useState([]); // [ { testId: string, testName: string, testDescription: string, testDuration: string, testQuestions: [ { questionId: string, questionText: string, questionOptions: [ { optionId: string, optionText: string, isCorrect: boolean } ] } ] }
  
  const handleCreateTaskClick = () => {
    setShowTestPage(true);
  };
  
  useEffect(() => {
    async function func() {
      if (starterKitIsConfiguredCorrectly) {
        authressLoginClient.userSessionExists().then((userIsLoggedIn) => {
          setUserProfile(authressLoginClient.getUserIdentity());
          console.log('User is Logged In', userIsLoggedIn, userProfile);
          console.log('User Profile', userProfile);
        });
      }
    }
   
    func();
  }, []);

  useEffect(() => {
    // Check if userProfile is not an empty object and if user is logged in
    if (userProfile) {
      // Make API call only if userProfile is available
      try {
        axios.post('http://localhost:5000/add-user', {
          userProfile
        }).then(response => {
          console.log('User added successfully:', response.data);
        }).catch(error => {
          console.error('Error adding user:', error);
        });
      } catch (error) {
        console.error('Error adding user:', error);
      }
    }

    if (userProfile) {
      try {
        console.log('Fetching tests for user:', userProfile);
        const data = axios.post('http://localhost:5000/all-exams', {
            userProfile
        }).then(response => {
          setTests(response.data);
          console.log('Tests fetched successfully:', response.data);
        }).catch(error => {
          console.error('Error fetching tests:', error);
        });
      } catch (error) {
          console.error('Error fetching tests:', error);
        }
    }

  }, [userProfile]);
  

  useEffect(() => {
    setAuthressApiUrlIsSet(starterKitIsConfiguredCorrectly);
  }, []);


  return (
    <>
    {!userProfile && 
      <button style={{ marginRight: '1rem' }} onClick={login}>
        Login
      </button>
    }
    {userProfile && (
      <>
      <ResponsiveAppBar userProfile = {userProfile} logout = {logout} setShowTestPage = {setShowTestPage} setShowAnalyticsPage = {setShowAnalyticsPage} setShowCreatePage = {setShowCreatePage} />
          <div style={{ display: 'flex', justifyContent: 'center', }}>


            {showTestPage && <CollapsibleTable tests = {tests} userProfile = {userProfile} />}

            {showCreatePage && <TestPage  />}

            {showAnalyticsPage && (
              <BasicLineChart />
            )}
          </div>
        </> 
    )}

    </>
  );
}

export default App;
