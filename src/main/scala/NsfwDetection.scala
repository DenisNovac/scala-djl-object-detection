import ai.djl.Application
import ai.djl.inference.Predictor
import ai.djl.modality.Classifications
import ai.djl.modality.cv.transform.{CenterCrop, Normalize, Resize, ToTensor}
import ai.djl.modality.cv.translator.ImageClassificationTranslator
import ai.djl.modality.cv.{Image, ImageFactory}
import ai.djl.repository.zoo.Criteria
import ai.djl.translate.Translator
import org.slf4j.LoggerFactory

import java.nio.file.Paths
import scala.jdk.CollectionConverters._

object NsfwDetection extends App {
  object Engines {

    object PyTorch {
      val name = "PyTorch"

      object Models {
        // https://huggingface.co/Falconsai/nsfw_image_detection 63e0a06
        // you need to run convert.py, it will download and automatically convert model to the needed one
        val falconsai = "converted.pt"
      }
    }

  }

  private val images = List(
    "dog_bike_car.jpg",
    "pony-toys.jpg",
    "hentai.jpg",
    "street.jpg",
    "nudity.jpg"
  )

  private val logger = LoggerFactory.getLogger("NsfwDetection")

  val translator: Translator[Image, Classifications] =
    ImageClassificationTranslator
      .builder()
      .optSynsetArtifactName("synset.txt")
      .addTransform(new Resize(256))
      // from the model description it was trained on 224x224 images
      .addTransform(new CenterCrop(224, 224))
      .addTransform(new ToTensor())
      .addTransform(
        new Normalize(
          Array(
            0.485f,
            0.456f,
            0.406f
          ),
          Array(
            0.229f,
            0.224f,
            0.225f
          )
        )
      )
      .optApplySoftmax(true)
      .build()

  val criteria: Criteria[Image, Classifications] = Criteria
    .builder()
    .setTypes(classOf[Image], classOf[Classifications])
    .optApplication(
      Application.CV.IMAGE_CLASSIFICATION
    )
    .optTranslator(
      translator
    )
    .optEngine(Engines.PyTorch.name)
    .optModelPath(Paths.get(Engines.PyTorch.Models.falconsai))
    .build();

  val model = criteria.loadModel();

  val predictor: Predictor[Image, Classifications] =
    model.newPredictor()

  images.map(predict)

  private def predict(imageName: String) = {
    val imagePath = Paths.get(s"images/$imageName")

    val img = ImageFactory.getInstance().fromFile(imagePath)

    /**  There is no point in keeping this structure so we'll consider an image nsfw if it's probability >0.7
      * [
      * {"class": "nsfw", "probability": 0.99987}
      * {"class": "normal", "probability": 0.00012}
      * ]
      */
    val detection = predictor.predict(img)

    val isNsfw =
      detection.getClassNames.asScala
        .zip(detection.getProbabilities.asScala)
        .toMap
        .get("nsfw")
        .exists(_ >= 0.7d)

    logger.info(s"$imageName -- nfsw: $isNsfw")

    detection
  }

  predictor.close()
  model.close()

}
