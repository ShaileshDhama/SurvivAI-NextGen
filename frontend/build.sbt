import org.scalajs.linker.interface.ModuleSplitStyle

lazy val frontend = project
  .in(file("."))
  .enablePlugins(ScalaJSPlugin)
  .settings(
    name := "survivai-frontend",
    version := "0.1.0",
    scalaVersion := "3.3.1",
    scalaJSUseMainModuleInitializer := true,
    Compile / fastLinkJS / scalaJSLinkerConfig ~= { 
      _.withModuleKind(ModuleKind.ESModule)
        .withModuleSplitStyle(ModuleSplitStyle.SmallModulesFor(List("survivai")))
    },
    libraryDependencies ++= Seq(
      "org.scala-js" %%% "scalajs-dom" % "2.6.0",
      "com.raquo" %%% "laminar" % "16.0.0",
      "com.github.japgolly.scalajs-react" %%% "core" % "2.1.1",
      "com.github.japgolly.scalajs-react" %%% "extra" % "2.1.1",
      "io.github.cquiroz" %%% "scala-java-time" % "2.5.0",
      "com.lihaoyi" %%% "upickle" % "3.1.3",
      "org.scala-js" %%% "scala-js-macrotask-executor" % "1.1.1"
    )
  )
