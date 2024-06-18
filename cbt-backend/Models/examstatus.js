const mongoose = require('mongoose');
const Schema = mongoose.Schema;


const UserTestSchema = new Schema({
  userId: { type: String, required: true },
  testId: { type: String, required: true },
  examStatus: [{}], // Using Map to store exam status for each topic
});

const UserTest = mongoose.model('UserTest', UserTestSchema);

module.exports = UserTest;