const mongoose = require('mongoose');

const testSchema = new mongoose.Schema({
  testname: { type: String },
  testduration: { type: Number },
  instructions: { type: String,},
  sections: [{}], 
  createdby: { type: String, required: true },
  ratings: { type: Number, required: true },
  difficulty: { type: String, required: true }
});

const Test = mongoose.model('Test', testSchema);

module.exports = Test;
