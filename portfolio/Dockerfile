FROM python:3.9

# Install necessary packages for extracting and configuring
RUN apt-get update && \
    apt-get install -y curl tar && \
    apt-get clean

# Copy the JDK tarball into the Docker image
COPY ./jdk-17.0.11_linux-x64_bin.tar.gz /tmp/jdk.tar.gz

# Extract the JDK tarball and set up JAVA_HOME
RUN mkdir -p /opt/java && \
    tar -xzf /tmp/jdk.tar.gz -C /opt/java && \
    mv /opt/java/jdk-17.0.11 /opt/java/jdk && \
    rm /tmp/jdk.tar.gz

# Copy Spark tarball into the Docker image
COPY ./spark-3.5.1-bin-hadoop3-scala2.13.tgz /tmp/spark.tgz

# Extract the Spark tarball and set up SPARK_HOME
RUN mkdir -p /opt/spark && \
    tar -xzf /tmp/spark.tgz -C /opt/spark && \
    mv /opt/spark/spark-3.5.1-bin-hadoop3-scala2.13 /opt/spark/spark && \
    rm /tmp/spark.tgz

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Set up environment variables
ENV JAVA_HOME="/opt/java/jdk"
ENV SPARK_HOME="/opt/spark/spark"
ENV PATH="${JAVA_HOME}/bin:${SPARK_HOME}/bin:/root/.cargo/bin:${PATH}"

USER root

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./ /code/scripts

WORKDIR /code/scripts

CMD ["streamlit", "run", "1_HomePage.py", "--server.port", "8081"]