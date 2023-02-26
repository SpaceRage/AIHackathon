document.getElementById("custom_image_submit").onclick = function () {
    console.log("clicked")
    var genderGroup = document.getElementsByName("gender_name")
    var male = document.getElementById("male")
    var female = document.getElementById("female")
    var image_upload = document.getElementById("image_upload");
    var sex = 1;
    if (female.checked) sex = 0
    // hide the button
    document.getElementById("custom_image_submit").style.opacity = 0;
    var localization = 6;
    var localValue = document.getElementById("location-select").value
    if (localValue === 'lower-ex') localization = 0
    else if (localValue === 'torso') localization = 1
    else if (localValue === 'upper-ex') localization = 2
    else if (localValue === 'head-neck') localization = 3
    else if (localValue === 'palm-sole') localization = 4
    else if (localValue === 'oral-genital') localization = 5
    else if (localValue === 'hand-foot') localization = 7
    else localization = 6

    const formData = new FormData();
    formData.append("file", image_upload.files[0]);
    formData.append("sex", sex);
    formData.append("age", Number(document.getElementById("age-input").value));
    formData.append("localization", localization);

    // Set up the options for the POST request
    const options = {
        method: "POST",
        headers: {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        },
        body: formData
    };
    

    // Send the POST request using fetch API
    var result = document.getElementById("result")

    result.innerHTML = "<br><br>" + "Processing..."
    fetch("http://127.0.0.1:5000/api/predict", options)
        .then((response) => {
            if (!response.ok) {
                throw new Error("Network response was not ok");
            }
            return response.json();
        })
        .then((responseBody) => {
            console.log(responseBody.data);
            if (Number(responseBody.data) == 0) {
                result.innerHTML = "<br><br>" + "Benign"
            } else {
                result.innerHTML = "<br><br>" + "Malignant"
            }
        })
        .catch((error) => {
            console.error("Error:", error);
            result.innerHTML = "<br><br>" + "An error occurred."
        });
        document.getElementById("custom_image_submit").style.opacity = 1;
}

// Preview the selected image
image_upload.onchange = (evt) => {
    const [file] = image_upload.files;
    if (file) {
        user_preview.src = URL.createObjectURL(file);
    }
};