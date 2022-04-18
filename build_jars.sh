cd gateway
gradle distZip
cd build/distributions
yes | unzip gateway.zip
cd ../../../
cp -r gateway/build/distributions/gateway/lib mowl
