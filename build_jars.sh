set -e
cd gateway
rm -f build/distributions/gateway/lib/*
gradle distZip
cd build/distributions
yes | unzip gateway.zip
cd ../../../
rm -f mowl/lib/*.jar
cp -r gateway/build/distributions/gateway/lib mowl
