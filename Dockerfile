FROM node:8.9.1
COPY . /app
WORKDIR /app
RUN \
npm install yarn -g --registry=https://registry.npm.taobao.org  && \
yarn install --force && \
EXPOSE 1234
CMD yarn watch