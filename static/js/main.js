let model = null;

async function getImage(img) {
    // let img = document.getElementById(id)
    
    img = tf.cast(tf.browser.fromPixels(img), 'float32');

    const offset = tf.scalar(255.0);
    // Normalize the image from [0, 255] to [0, 1].
    img = img.div(offset)
    return img
}


async function stylize(content_elem, style_elem, result_elem) {
    try {
        // console.log(tf.getBackend());
        const content = await getImage(content_elem)
        const style = await getImage(style_elem)
        const iters = tf.scalar(
            // document.getElementById('iters').value, 
            1,
            dtype='int32')
        const max_resolution = tf.scalar(
            // document.getElementById('max_resolution').value,
            256,
            dtype='int32')
        
        console.log(content, style);
        console.log('Images loaded successfully!');

        const model = await loadModel();

        console.log('Model loaded successfully!');
        
        console.log(model);
        console.log('Doing stylization...');
        const result = await model.executeAsync(
            {content,style,iters,max_resolution},
            ["result"]
        ); 
        console.log(result);
        await tf.browser.toPixels(result, result_elem);
    } catch (e) {
        console.error(e);
    }
}

async function load_tfjs_model() {
    console.log("Loading model...");
    model = await tf.loadGraphModel('static/js/model/model.json');
    console.log("Model loaded.");
    console.log('Warming up model...');
    console.log(model);
    // tf.enableDebugMode()
    await model.executeAsync(
        {
            'image':tf.ones([64,64,3], dtype ='float32'),
        },
    ); 
    console.log('Model warmed up.');
}

window.addEventListener("load", e => {
    console.log("window loaded");
    load_tfjs_model();
});

document.getElementById('submit').addEventListener('click', 
    e => {
        console.log("submit clicked");
    }
);

