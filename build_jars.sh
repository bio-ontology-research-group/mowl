set -e
# Rebuilds mowl/lib: gateway.jar, jcel-owlapi5 and the libraries resolved from Maven
# Central (see gateway/build.gradle). The KM reasoner JAR is kept.
cd gateway
./gradlew --no-daemon clean installLibs
