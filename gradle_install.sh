wget https://services.gradle.org/distributions/gradle-7.4.2-bin.zip -P /tmp

unzip -d /tmp/gradle /tmp/gradle-*.zip
export GRADLE_HOME=/tmp/gradle/gradle-7.4.2
export PATH=${GRADLE_HOME}/bin:${PATH}

