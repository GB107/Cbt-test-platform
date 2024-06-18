const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const userSchema = new Schema({
    id: { type: String, required: true },
    name: { type: String, required: true },
    email: { type: String, required: true },
    photo : { type: String, required: true },
    tests: [{}], 

 });
 
const Users = mongoose.model('cbtusers',userSchema);

module.exports = Users;