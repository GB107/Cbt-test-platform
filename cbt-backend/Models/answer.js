const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const answerSchema = new Schema({
    test: {
        type: Schema.Types.ObjectId,
        ref: 'Test', 
        required: true
    },
    answered_by: {
        type: Schema.Types.ObjectId,
        ref: 'User', 
        required: true
    },
    answers: {
        type: Object,
        required: true
    },
    // status: {
    //     type: String,
    //     enum: ['answered', 'marked', 'reviewed'], // Enumerating possible status values
    //     required: true
    // }
});

module.exports = mongoose.model('Answer', answerSchema);
