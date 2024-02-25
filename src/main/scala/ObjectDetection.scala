import ai.djl.Application
import ai.djl.engine.Engine
import ai.djl.modality.cv.output.DetectedObjects
import ai.djl.modality.cv.{Image, ImageFactory}
import ai.djl.repository.zoo.{Criteria, ModelZoo, ZooModel}
import ai.djl.training.util.ProgressBar
import org.slf4j.LoggerFactory

import java.nio.file.{Files, Paths}
import java.time.Instant
import scala.jdk.CollectionConverters._

object ObjectDetection extends App {
  object Engines {

    object MxNet {
      val name = "MXNet"

      object Models {
        val darknet53 = "darknet53"
        val mobileNet = "mobilenet1.0"
        val resnet50 = "resnet50"
        val vgg16 = "vgg16"
      }
    }
  }

  private val images =
    List(
      "dog_bike_car.jpg",
      "pony-toys.jpg",
      "street.jpg"
    )

  private val logger = LoggerFactory.getLogger("ObjectDetection")

  private val models = ModelZoo
    .listModels()
    .asScala
    .map(t => t._1 -> t._2.asScala.mkString("\n"))
    .mkString("\n\n")

  println("Available models for engine: " + models)

  private val useEngine = Engines.MxNet.name
  private val useModel = Engines.MxNet.Models.vgg16

  private val criteria: Criteria[Image, DetectedObjects] =
    Criteria.builder
      .optApplication(Application.CV.OBJECT_DETECTION)
      .setTypes(classOf[Image], classOf[DetectedObjects])
      .optEngine(Engine.getEngine(useEngine).getEngineName)
      .optFilter("backbone", useModel)
      .optProgress(new ProgressBar)
      .build

  private val model: ZooModel[Image, DetectedObjects] = criteria.loadModel()

  private val predictor = model.newPredictor()

  images.map(predict)

  private def predict(imageName: String) = {
    val imagePath = Paths.get(s"images/$imageName")

    val img = ImageFactory.getInstance().fromFile(imagePath)

    val detection = predictor.predict(img)

    if (detection.getNumberOfObjects == 0)
      logger.warn(s"Nothing was detected on $imageName")
    else {
      logger.info(s"$imageName: $detection")
      saveBounds(img, imageName, detection)
    }

    detection
  }

  private def saveBounds(
      image: Image,
      imageName: String,
      detection: DetectedObjects
  ): Unit = {
    val fmt = "png"

    val ts = Instant.now().getEpochSecond

    val resultName =
      imageName
        .split("\\.")
        .dropRight(1)
        .mkString("") + s"-$useEngine-$useModel-$ts.$fmt".toLowerCase

    val imagePath = Paths.get(s"output/$resultName")

    image.drawBoundingBoxes(detection)

    image.save(Files.newOutputStream(imagePath), fmt)

    logger.info(s"Detected objects image has been saved in $imagePath")
  }

  predictor.close()
  model.close()
}
