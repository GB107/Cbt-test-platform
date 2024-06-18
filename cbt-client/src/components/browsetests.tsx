import * as React from 'react';
import PropTypes from 'prop-types';
import Box from '@mui/material/Box';
import Collapse from '@mui/material/Collapse';
import IconButton from '@mui/material/IconButton';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';
import KeyboardArrowUpIcon from '@mui/icons-material/KeyboardArrowUp';
import { Button } from '@mui/material';
import axios from 'axios';

function createData(testId, name, calories, fat, carbs, protein,  history) {
  return {
    testId,
    name,
    calories,
    fat,
    carbs,
    protein,
    history: [
      history
    ],
  };
}



function Row(props) {

  const { row } = props;
  const [open, setOpen] = React.useState(false);
  const [data, setData] = React.useState([]);



  return (
    <React.Fragment>
      <TableRow sx={{ '& > *': { borderBottom: 'unset' } }}>
        <TableCell>
          <IconButton
            aria-label="expand row"
            size="small"
            onClick={() => setOpen(!open)}
          >
            {open ? <KeyboardArrowUpIcon /> : <KeyboardArrowDownIcon />}
          </IconButton>
        </TableCell>
        <TableCell component="th" scope="row">
          {row.name}
        </TableCell>
        <TableCell align="right">{row.calories}</TableCell>
        <TableCell align="right">{row.fat}</TableCell>
        <TableCell align="right">{row.carbs}</TableCell>
        <TableCell align="right">{row.protein}</TableCell>
      </TableRow>
      <TableRow>
        <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={6}>
          <Collapse in={open} timeout="auto" unmountOnExit>
            <Box sx={{ margin: 1 }}>
              <Typography variant="h6" gutterBottom component="div">
                Test Details
              </Typography>
              <Table size="small" aria-label="purchases">
                <TableHead>
                  <TableRow>
                    <TableCell>Section Name</TableCell>
                    <TableCell>No. of Questions</TableCell>
                    <TableCell align="right">Positive Marks</TableCell>
                    <TableCell align="right">Negative Marks</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  
                  {row.history.map((historyRow) => (
                    <>
                    <TableRow key={historyRow[0].sectionName}>
                      <TableCell component="th" scope="row">
                        {historyRow[0].sectionName}
                      </TableCell>
                      <TableCell>{historyRow[0].totalQuestions}</TableCell>
                      <TableCell align="right">{historyRow[0].positiveMarks}</TableCell>
                      <TableCell align="right">
                        {historyRow[0].negativeMarks === -1 ? "No Negative Marking" : "-" + historyRow[0].negativeMarks}
                      </TableCell>
                    </TableRow>
                    </>
                  ))}
                </TableBody>
              </Table>
              <Box
                               sx={
                                  {
                                    display: "flex",
                                    flexDirection: "row",
                                    justifyContent: "flex-end",
                                    m: 2
                                  }
                                    
                                }
              >
              <Button

                variant="contained"
                color='success'
                onClick={() => {
                  console.log("http://localhost:5173/?testId=" + 
                  row.testId + "&userId=" +
                  props.userProfile.userId + "&name=" + props.userProfile.name + "&photo=" + props.userProfile.picture)

                  window.open("http://localhost:5173/?testId=" + 
                                    row.testId + "&userId=" +
                                    props.userProfile.userId + "&name=" + props.userProfile.name + "&photo=" + props.userProfile.picture)
                }}
 
              >
                Start Test
              </Button>
            </Box>
            </Box>
          </Collapse>
        </TableCell>
      </TableRow>
    </React.Fragment>
  );
}

Row.propTypes = {
  row: PropTypes.shape({
    calories: PropTypes.number.isRequired,
    carbs: PropTypes.number.isRequired,
    fat: PropTypes.number.isRequired,
    history: PropTypes.arrayOf(
      PropTypes.shape({
        amount: PropTypes.number.isRequired,
        customerId: PropTypes.string.isRequired,
        date: PropTypes.string.isRequired,
      }),
    ).isRequired,
    name: PropTypes.string.isRequired,
    price: PropTypes.number.isRequired,
    protein: PropTypes.number.isRequired,
  }).isRequired,
};

// const rows = [
//   createData('Frozen yoghurt', 159, 6.0, 24, 4.0, 3.99),
//   createData('Ice cream sandwich', 237, 9.0, 37, 4.3, 4.99),
//   createData('Eclair', 262, 16.0, 24, 6.0, 3.79),
//   createData('Cupcake', 305, 3.7, 67, 4.3, 2.5),
//   createData('Gingerbread', 356, 16.0, 49, 3.9, 1.5),
// ];


export default function CollapsibleTable(props) {

  const [tests, setTests] = React.useState([]);
  const [rows, setRows] = React.useState([]);
  
  React.useEffect(() => {
    if (props.tests) {
      setTests(props.tests.tests);
    }
}, [props.tests]);

    React.useEffect(() => {
      if (tests) {
        // rows.push(createData(test.testName, test.totalQuestions, test.totalMarks, test.rating, test.difficulty, test.sections));

        let rows = [];
        tests.forEach(test => {
          rows.push(createData(test.test_id, test.testname, test.totalQuestions,  "100", test.ratings, test.difficulty, test.sectionsWiseData));
        });
        setRows(rows);

        console.log(rows)
      }
    }, [tests]);

  return (
    <TableContainer component={Paper}
      sx={{
        m: 5
      }}
    >
      <Table aria-label="collapsible table">
        <TableHead>
          <TableRow>
            <TableCell />
            <TableCell>Test Name</TableCell>
            <TableCell align="right">Total Questions</TableCell>
            <TableCell align="right">Total Marks</TableCell>
            <TableCell align="right">Rating (10)</TableCell>
            <TableCell align="right">Difficulty (100)</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map((row, idx) => (
            <Row key={row.name} row={row} idx = {idx} userProfile = {props.userProfile} />
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}