// custom.js

// Function to periodically update the text content, image source, and translation content
function updateContent() {
    $.ajax({
        url: "/get_text",  // Flask route to get the detected text
        type: "GET",
        success: function (data) {
            updateTextContent(data);  // Update the text content
            updateImageSource();      // Update the image source

            // Translate the text and update the translation content
            $.ajax({
                url: "/translate_text",  // Flask route to translate text
                type: "POST",
                data: { text: data },
                success: function (translation) {
                    updateTranslationContent(translation);  // Update the translation content
                },
                error: function () {
                    // Handle error if translation fails
                }
            });

            setTimeout(updateContent, 1000);  // Update every 1000 milliseconds (1 second)
        },
        error: function () {
            setTimeout(updateContent, 1000);
        }
    });
}

// Start updating content when the page is loaded
$(document).ready(function () {
    updateContent();
});

// Function to update the text content
function updateTextContent(text) {
    var lines = text.split('\n');
    $("#text_output").empty();
    lines.forEach(function(line) {
        $("#text_output").append("<div>" + line + "</div>");
    });
}

// Function to update the translation content
function updateTranslationContent(text) {
    var lines = text.split('\n');
    $("#translation_output").empty();
    lines.forEach(function(line) {
        $("#translation_output").append("<div>" + line + "</div>");
    });
}

// Function to update the image source
function updateImageSource() {
    $("#video_feed").attr("src", "{{ url_for('video_feed') }}?" + new Date().getTime());
}
