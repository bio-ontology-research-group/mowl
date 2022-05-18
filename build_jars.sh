cd gateway
gradle distZip
cd build/distributions
yes | unzip gateway.zip
cd ../../../
rm -f mowl/lib/*.jar
touch mowl/lib/aux
cp -r gateway/build/distributions/gateway/lib mowl
