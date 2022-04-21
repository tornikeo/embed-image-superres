let MODEL = null;

async function getImage(img) {
    // let img = document.getElementById(id)
    

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
    tf.loadGraphModel('static/js/model/model.json')
    .then(model => {
        console.log("Model loaded.");
        console.log('Warming up model...');
        console.log(model);
        MODEL = model;

    //     return model.executeAsync(
    //         {
    //             'image':tf.ones([64,64,3], dtype ='float32'),
    //         },
    //     )
    // })
    // .then(model => {
    //     console.log('Model warmed up.');
    //     console.log(model);
    //     console.log('Model ready!');
    //     MODEL = model;
    });
}

window.addEventListener("load", e => {
    console.log("window loaded");
    load_tfjs_model();
});

async function predict(img) {
    if (MODEL !== null) {
        console.log("Predicting...");
        let result = await MODEL.executeAsync(
            {
                'image':img,
            },
            // ["result"]
        ); 
        result = tf.clipByValue(result, 0, 1);
        console.log(result);
        return result;
    }
}

function displayResult(img) {
    tf.browser.toPixels(img, document.getElementById('result'));
}

async function loadImage() {
    img = document.getElementById('imageResult');
    img = tf.browser.fromPixels(img)
    img = tf.div(img,255.)
    img = tf.cast(img, 'float32');
    console.log(img);
    return img
}

document.getElementById('submit').addEventListener('click', 
    e => {
        console.log("submit clicked");
        loadImage()
            .then(img => {
                return predict(img)
            })
            .then(result => {
                console.log(result);
                displayResult(result);
            })
            .catch(e => {
                console.error(e);
            });
    }
);

