import * as React from 'react';
import { LineChart } from '@mui/x-charts/LineChart';
import { Box, Typography } from '@mui/material';
import { PieChart } from '@mui/x-charts/PieChart';

export default function BasicLineChart() {
  // Sample data from API response
  const apiData = [
    {
      "testName": "Test1",
      "testId": "65ececfe181e16a1aaf83db0",
      "counts": {
        "na": 2,
        "nv": 0,
        "amr": 1,
        "mr": 0,
        "an": 2
      }
    },
    {
      "testName": "Test2",
      "testId": "65ececfe181e16a1aaf83db0",
      "counts": {
        "na": 2,
        "nv": 0,
        "amr": 1,
        "mr": 0,
        "an": 2
      }
    }
  ];

  const keyToLabel = {
    "na": "Not Answered",
    "nv": "Not Visited",
    "amr": "Answered Marked for Review",
    "mr": "Marked for Review",
    "an": "Answered"
  };

  // Extract relevant data for the line chart
  const lineChartData = {
    //number from 1 to apiData.length
    xAxis: [{ data: apiData.map((item, index) => index + 1) }],
    series: [{
      data: apiData.map(item => Object.values(item.counts).reduce((acc, curr) => acc + curr, 0)) // Total questions attempted for each test
    }]
  };

  // Extract relevant data for the pie chart
  const pieChartData = apiData.map(item => ({
    data: Object.entries(item.counts).map(([key, value]) => ({ id: key, value, label: keyToLabel[key] })),
    testName: item.testName
  }));

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <Box
        sx={{
          backgroundColor: 'white',
          m: 3,
          borderRadius: 2,
          boxShadow: 3,
        }}
      >
        <Typography variant="h6" component="div" sx={{ m: 2, color: 'black' }}>
          Total Attempted v/s Total Questions
        </Typography>
        <LineChart
          xAxis={lineChartData.xAxis}
          series={lineChartData.series}
          width={500}
          height={300}
        />
      </Box>
      {pieChartData.map(({ testName, data }, index) => (
        <Box
          key={index}
          sx={{
            backgroundColor: 'white',
            m: 3,
            borderRadius: 2,
            boxShadow: 3,
            width: 900,
            display: 'flex',
            justifyContent: 'center',
            flexDirection: 'column',
          }}
        >
          <Typography variant="h6" component="div" sx={{ m: 2, color: 'black' }}>
            Question Attempt Status for {testName}
          </Typography>
          <PieChart
            key={index}
            series={[{ 
                data,
                highlightScope: { faded: 'global', highlighted: 'item' },
                faded: { innerRadius: 30, additionalRadius: -30, color: 'gray' },
            }]}
            height={200}
            sx={{ width: "90%" }}
            
          />
        </Box>
      ))}
    </Box>
  );
}
