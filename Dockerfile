FROM node:8.9.1
COPY . /app
WORKDIR /app
RUN \
npm install yarn -g --registry=https://registry.npm.taobao.org  && \
yarn install --force && \
yarn global add cross-env
EXPOSE 1234
CMD yarn watch