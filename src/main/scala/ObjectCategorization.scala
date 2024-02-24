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

object ObjectCategorization extends App {
  object Engines {

    object PyTorch {
      val name = "PyTorch"

      object Models {
        // https://huggingface.co/Falconsai/nsfw_image_detection
        val falconsai = "converted.pt"
      }
    }

  }

  private val imageName = "dog_bike_car.jpg"

  private val logger = LoggerFactory.getLogger("ObjectDetection")

  predict()

  private def predict() = {
    val imagePath = Paths.get(s"images/$imageName")

    val img = ImageFactory.getInstance().fromFile(imagePath)

    val translator: Translator[Image, Classifications] =
      ImageClassificationTranslator
        .builder()
        .optSynsetArtifactName("synset.txt")
        .addTransform(new Resize(256))
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

    val detection = predictor.predict(img)

    predictor.close()
    model.close()

    logger.info(s"Detection: $detection")

    detection
  }

}
