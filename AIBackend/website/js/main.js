document.getElementById("custom_image_submit").onclick = function () {
    var genderGroup = document.getElementsByName("gender_name")
    var checkedSex = Array.from(genderGroup).find((sex) => sex.checked)

    var sex = 1;
    if (sex.checked == "female") sex = 0;

    var localization = 6;
    var localValue = document.getElementById("location-select").value
    if (localValue == 'lower-ex') localization = 0
    else if (localValue == 'torso') localization = 1
    else if (localValue == 'upper-ex') localization = 2
    else if (localValue == 'head-neck') localization = 3
    else if (localValue == 'palm-sole') localization = 4
    else if (localValue == 'oral-genital') localization = 5
    else if (localValue == 'hand-foot') localization = 7
    else localization = 6
    

    // Set up the formData object
    const formData = new FormData();
    formData.append("file", image_upload.files[0]);
    formData.append("sex", sex);
    formData.append("age", Number(document.getElementById("age-input").value));
    formData.append("localization", 2);

    // Set up the options for the POST request
    const options = {
        method: "POST",
        url: "http://127.0.0.1:5000/model",
        headers: {},
        body: formData,
    };

    // Send the POST request using fetch API
    fetch(options.url, {
        method: options.method,
        body: options.body,
    })
        .then((response) => {
            if (!response.ok) {
                throw new Error("Network response was not ok");
            }
            return response.text();
        })
        .then((responseBody) => {
            console.log(responseBody);
        })
        .catch((error) => {
            console.error("Error:", error);
        });
}

// Preview the selected image
image_upload.onchange = (evt) => {
  const [file] = image_upload.files;
  if (file) {
    user_preview.src = URL.createObjectURL(file);
  }
};

