// var request = require('request');
// var fs = require('fs');

import request from "request"
import fs from "fs"
var options = {
    'method': 'POST',
    'url': 'http://127.0.0.1:5000/model',
    'headers': {
    },
    formData: {
        'file': {
            'value': fs.createReadStream(user_preview.src),
            'options': {
                'filename': 'uploaded.jpg',
                'contentType': null
            }
        }
    }
};
request(options, function (error, response) {
    if (error) throw new Error(error);
    console.log(response.body);
});


image_upload.onchange = evt => {
    const [file] = image_upload.files
    if (file) {
        user_preview.src = URL.createObjectURL(file)
    }
}

