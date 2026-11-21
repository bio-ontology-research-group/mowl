rem Rebuilds mowl\lib: gateway.jar, jcel-owlapi5 and the libraries resolved from Maven
rem Central (see gateway\build.gradle). The KM reasoner JAR is kept.
cd gateway
gradlew.bat --no-daemon clean installLibs
cd ..
