#!/usr/bin/env bash
# Rebuild the JAR files in mowl/lib (OWL API 5.5.1, ELK 0.6.0, HermiT 1.4.5, mOWL's gateway).
#
# Needs a JDK (11 or later), Gradle 8, curl and unzip.
#
# Two JAR files are not taken from Maven Central as they are:
#
# * jcel-owlapi: the released jcel-owlapi 0.24.1 is compiled against OWL API 4 and fails to
#   link with OWL API 5. Its sources compile unchanged against OWL API 5, so this script
#   downloads the 0.24.1 sources and recompiles them (jcel-owlapi-0.24.1-owlapi5.jar).
# * kobayashi-marust-owlapi5-<version>.jar: the KM reasoner binding built for OWL API 5
#   (see https://github.com/bio-ontology-research-group/kobayashi-marust). It is kept as it
#   is; this script does not rebuild it.
set -e
JCEL_VERSION=0.24.1
MAVEN=https://repo1.maven.org/maven2

cd gateway
rm -r -f build/
gradle distZip
cd build/distributions
unzip -q -o gateway.zip
cd ../../../

find mowl/lib -name '*.jar' ! -name 'kobayashi-marust-*' -delete
cp gateway/build/distributions/gateway/lib/*.jar mowl/lib/

# jcel-owlapi, recompiled against the OWL API that was just installed
work=gateway/build/jcel-owlapi
rm -r -f "$work"
mkdir -p "$work/src" "$work/classes"
curl -s -S -f -o "$work/sources.jar" \
    "$MAVEN/de/tu-dresden/inf/lat/jcel/jcel-owlapi/$JCEL_VERSION/jcel-owlapi-$JCEL_VERSION-sources.jar"
unzip -q "$work/sources.jar" -d "$work/src"
rm -f "mowl/lib/jcel-owlapi-$JCEL_VERSION.jar"
classpath=$(find mowl/lib -name '*.jar' | tr '\n' ':')
find "$work/src" -name '*.java' > "$work/sources.txt"
javac -proc:none -nowarn --release 11 -d "$work/classes" -cp "$classpath" "@$work/sources.txt"
printf 'Implementation-Title: jcel-owlapi\nImplementation-Version: %s\nImplementation-URL: https://github.com/julianmendez/jcel\nComment: unmodified jcel-owlapi sources, recompiled against OWL API 5\n' \
    "$JCEL_VERSION" > "$work/manifest.txt"
jar cfm "mowl/lib/jcel-owlapi-$JCEL_VERSION-owlapi5.jar" "$work/manifest.txt" -C "$work/classes" .
