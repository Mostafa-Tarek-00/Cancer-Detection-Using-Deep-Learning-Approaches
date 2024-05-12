document.addEventListener("DOMContentLoaded", function() {
    const predictButton = document.getElementById("predict-button");
    const imageUpload = document.getElementById("image-upload");
    const uploadedImage = document.getElementById("uploaded-image");
    const predictionResult = document.getElementById("prediction-result");
    const floatingWindow = document.getElementById("floating-window");
    const floatingContent = document.getElementById("floating-content");

    // Treatment suggestions for each class
    const treatmentSuggestions = {
        'Melanocytic nevi (nv)': 'For Melanocytic nevi (nv), no treatment is usually required. However, it is recommended to monitor for any changes in size, shape, or color.',
        'Melanoma (mel)': 'For Melanoma (mel), treatment options may include surgery, chemotherapy, immunotherapy, or targeted therapy. Consult a dermatologist for proper evaluation and treatment.',
        'Benign keratosis-like lesions (bkl)': 'For Benign keratosis-like lesions (bkl), treatment may involve cryotherapy, curettage, or topical medications. Consult a dermatologist for further evaluation.',
        'Basal cell carcinoma (bcc)': 'For Basal cell carcinoma (bcc), treatment options may include surgery, radiation therapy, or topical medications. Consult a dermatologist for proper evaluation and treatment.',
        'Actinic keratoses (akiec)': 'For Actinic keratoses (akiec), treatment options may include cryotherapy, photodynamic therapy, or topical medications. Consult a dermatologist for further evaluation.',
        'Vascular lesions (vasc)': 'For Vascular lesions (vasc), treatment may involve laser therapy, sclerotherapy, or surgical removal. Consult a dermatologist for proper evaluation and treatment.',
        'Dermatofibroma (df)': 'For Dermatofibroma (df), treatment is usually not required unless the lesion causes discomfort or cosmetic concerns. In such cases, surgical excision may be performed. Consult a dermatologist for proper evaluation.'
    };

    predictButton.addEventListener("click", function() {
        imageUpload.click();
    });

    imageUpload.addEventListener("change", function() {
        const file = imageUpload.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                uploadedImage.src = e.target.result;
                predictImage(file);
            };
            reader.readAsDataURL(file);
        }
    });

    function predictImage(file) {
        const formData = new FormData();
        formData.append("image", file);
    
        fetch("/predict", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data && data.overlay_image) {
                const overlayImageUrl = data.overlay_image;
                showOverlayImage(overlayImageUrl);
            } else if (data && data.prediction) {
                const predictionClass = data.prediction;
                const confidence = Math.round(data.confidence * 100);
                const predictionText = `Prediction: ${predictionClass}, Confidence: ${confidence}%`;
                const treatmentSuggestion = treatmentSuggestions[predictionClass];
                if (treatmentSuggestion) {
                    const combinedText = `${predictionText}<br>Treatment Suggestion: ${treatmentSuggestion}`;
                    predictionResult.innerHTML = combinedText;
                    floatingContent.innerHTML = combinedText;
                    floatingWindow.style.display = "block";
                } else {
                    const genericSuggestion = "No specific treatment suggestion available. Please consult a dermatologist for proper evaluation and treatment advice.";
                    const combinedText = `${predictionText}<br>${genericSuggestion}`;
                    predictionResult.innerHTML = combinedText;
                }
            } else {
                predictionResult.innerHTML = "Error: Unable to make prediction or no treatment suggestion available.";
            }
        })
        .catch(error => {
            console.error("Error:", error);
            predictionResult.innerHTML = "Error: Unable to make prediction.";
        });
    }
    
    
    function showOverlayImage(overlayImageUrl) {
        // Replace the uploaded image with the overlay image
        uploadedImage.src = overlayImageUrl;
    
        // Hide the prediction result div
        predictionResult.innerHTML = "";
    
        // Display the uploaded image with the overlay
        uploadedImage.style.opacity = 0.7; // Set opacity to see both original and overlay
    }

    // Close floating window
    const closeButton = document.getElementById("close-button");
    closeButton.addEventListener("click", function() {
        floatingWindow.style.display = "none";
    });
});
