const classifier = knnClassifier.create()
const webcamElement = document.getElementById("webcam")

let net

async function app() {

    console.log("Loading mobilnet...")

    net = await mobilenet.load()

    console.log("Model Loaded")

    const webcam = await tf.data.webcam(webcamElement)

    const addExample = async (classId) => {
        const img = await webcam.capture()

        const activation = net.infer(img, true)

        classifier.addExample(activation, classId)

        img.dispose()
    }

    document.getElementById("dog").addEventListener("click", () => addExample(0))
    document.getElementById("cat").addEventListener("click", () => addExample(1))
    document.getElementById("snake").addEventListener("click", () => addExample(2))
    document.getElementById("spider").addEventListener("click", () => addExample(3))
    document.getElementById("fish").addEventListener("click", () => addExample(4))

    while(true) {
        
        if(classifier.getNumClasses() > 0){
            const img = await webcam.capture()

            const activation = net.infer(img, "conv_preds")

            const result = await classifier.predictClass(activation)

            const classes = ["Dog", "Cat", "Snake", "Spider", "Fish"]

            document.getElementById("console").innerText = `
                prediction: ${classes[result.label]}\n
                probability: ${result.confidences[result.label]} 
            `

            img.dispose()
        }

        await tf.nextFrame()
    }
}

app()