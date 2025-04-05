FROM gcr.io/dataflow-templates-base/flex-template-launcher-image:latest AS template_launcher

FROM python:3.10-slim

WORKDIR /workspace

COPY requirements.txt requirements.txt

ENV FLEX_TEMPLATE_PYTHON_REQUIREMENTS_FILE="/workspace/requirements.txt"
ENV FLEX_TEMPLATE_PYTHON_PY_FILE="/workspace/df_flex/main.py"
ENV FLEX_TEMPLATE_PYTHON_SETUP_FILE="/workspace/setup.py"

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl https://bootstrap.pypa.io/get-pip.py | python \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt 

# Copy the run module
COPY ./df_flex/ /workspace/df_flex/
COPY setup.py /workspace
RUN rm -fr /workspace/__pycache__

# Since we already downloaded all the dependencies, there's no need to rebuild everything.
ENV PIP_NO_DEPS=True

ENV PYTHONPATH="/workspace:${PYTHONPATH}"

# Copy the Dataflow Template launcher
COPY --from=template_launcher /opt/google/dataflow/python_template_launcher /opt/google/dataflow/python_template_launcher

# Copy files from official SDK image, including script/dependencies.
COPY --from=apache/beam_python3.10_sdk:latest /opt/apache/beam /opt/apache/beam

# Set the entrypoint to Apache Beam SDK launcher.
ENTRYPOINT ["/opt/apache/beam/boot"]
