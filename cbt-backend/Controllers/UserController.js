const UsersModel = require('../Models/users.js');
const jwt = require('jsonwebtoken');
class userController{
    static AddUser = (req, res) => {
        const dis = this;
        // console.log(req.body.userProfile);
        const userProfile = req.body.userProfile; 
    
        if (!userProfile || !userProfile.userId || !userProfile.name || !userProfile.email || !userProfile.picture) {
            return res.status(400).send({
                isSuccessful: false,
                message: "User profile data is incomplete"
            });
        }
        const userId = userProfile.userId; 
        UsersModel.findOne({ id: userId }) 
            .then(existingUser => {
                if (existingUser) {
                    return res.status(400).send({
                        isSuccessful: false,
                        message: "User already exists"
                    });
                } else {
                    // User with given userId does not exist, create and save
                    const userData = {
                        id: userProfile.userId,
                        name: userProfile.name,
                        email: userProfile.email,
                        photo: userProfile.picture,
                    };
                    const user = new UsersModel(userData);
                    user.save()
                        .then(() => {
                            dis.FindUsers(res);
                        })
                        .catch((err) => {
                            console.log(err);
                            res.status(500).send({
                                isSuccessful: false,
                                message: "An error occurred while saving the user"
                            });
                        });
                }
            })
            .catch(err => {
                console.log(err);
                res.status(500).send({
                    isSuccessful: false,
                    message: "An error occurred while checking user existence"
                });
            });
    }
    
    

    static FindUsers = (res)=>{
        UsersModel.find()
        .then((result)=>{
            res.send({
                isSuccessful: true,
                data:result
            });
        })
        .catch((err)=>{
            console.log(err);
        })
    }
    static FindSingleUser = (req,res)=>{
        UsersModel.findById(req.body)
        .then((result)=>{
            res.send(result)
        })
        .catch((err)=>{
            console.log(err);
        })
    }
    static FindUser = (req,res)=>{
        UsersModel.findOne(req.body.query)
        .then((result)=>{
            jwt.sign({user:result}, 'secretkey', {expiresIn: '600'}, (err,token)=>{
                res.send({
                    isSuccessful: true,
                    token: token
                })
            }) 
        })
        .catch((err)=>console.log(err))  
    }
    static FindOneUser = (req,res)=>{
        UsersModel.findOne(req.body.query)
        .then((result)=>{
            res.send({
                isSuccessful: true,
                data:result
            });
        })
        .catch((err)=>{
            console.log(err);
        })
    }
 
}

module.exports = userController;