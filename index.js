/*
 * @Descripttion: 
 * @version: 1.0.0
 * @Author: 孙立政
 * @Date: 2023-08-17 21:23:01
 * @LastEditors: 孙立政
 * @LastEditTime: 2023-08-17 21:34:56
 */
const express = require("express");
const axios = require('axios');
const tf = require('@tensorflow/tfjs-node');
const mobilenet = require('@tensorflow-models/mobilenet');

const app = express();

app.all('*', function (req, res, next) {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Headers", "X-Requested-With");
  res.header("Access-Control-Allow-Methods", "PUT,POST,GET,DELETE,OPTIONS");
  res.header("X-Powered-By", '3.2.1')
  res.header("Content-Type", "application/json;charset=utf-8");
  next();
});


const bodyParser = require('body-parser');
app.use(bodyParser.json({ limit: '9mb' }));
app.use(bodyParser.urlencoded({ extended: true }));


app.get('/img2embedding', async function (req, res) {
  try {
    const imageURL = req.query.url;

    const response = await axios.get(imageURL, { responseType: 'arraybuffer' });
    const imageBuffer = Buffer.from(response.data, 'binary');

    const decodedImage = tf.node.decodeImage(imageBuffer);
    const resizedImage = tf.image.resizeBilinear(decodedImage, [224, 224]);
    const batchedImage = resizedImage.expandDims(0);

    const model = await mobilenet.load();
    const embedding = await model.infer(batchedImage, { logits: true });
    const embeddingArray = embedding.arraySync();

    res.json({
      status: 200, 
      message: 'ok', 
      data: {
        url: req.query.url,
        emm: embeddingArray[0]
      }
    });
  } catch (error) { 
    res.json({ 
      status: 500, 
      message: error.message, 
    });
  }
});


app.listen(80, () => { })