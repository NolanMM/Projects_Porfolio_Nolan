#!/bin/bash

# Install Java
echo "Installing Java..."
sudo apt-get update
sudo apt-get install -y openjdk-8-jdk

# Set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

# Install Apache Spark
echo "Installing Apache Spark..."
wget https://downloads.apache.org/spark/spark-3.5.1/spark-3.5.1-bin-hadoop3-scala2.13.tgz -P /tmp
tar -xvzf /tmp/spark-3.5.1-bin-hadoop3-scala2.13.tgz -C /opt
ln -s /opt/spark-3.5.1-bin-hadoop3-scala2.13 /opt/spark
rm /tmp/spark-3.5.1-bin-hadoop3-scala2.13.tgz

# Set SPARK_HOME
export SPARK_HOME=/opt/spark

# Add Spark to PATH
export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

# Install Apache Cassandra
echo "Installing Apache Cassandra..."
echo "deb http://www.apache.org/dist/cassandra/debian 311x main" | sudo tee -a /etc/apt/sources.list.d/cassandra.sources.list
curl https://www.apache.org/dist/cassandra/KEYS | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y cassandra

# Set CASSANDRA_HOME
export CASSANDRA_HOME=/usr/share/cassandra

# Add Cassandra to PATH
export PATH=$PATH:$CASSANDRA_HOME/bin

# Start Cassandra service
sudo service cassandra start

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup completed!"
