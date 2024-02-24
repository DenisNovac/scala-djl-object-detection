ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.12"

lazy val root = (project in file("."))
  .settings(
    name := "image-recognition-ex"
  )

libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.4.14"

libraryDependencies += "ai.djl" % "api" % "0.26.0"

libraryDependencies += "ai.djl.mxnet" % "mxnet-model-zoo" % "0.26.0"
libraryDependencies += "ai.djl.mxnet" % "mxnet-engine" % "0.26.0"

libraryDependencies += "ai.djl.pytorch" % "pytorch-engine" % "0.26.0"
libraryDependencies += "ai.djl.pytorch" % "pytorch-model-zoo" % "0.26.0"
