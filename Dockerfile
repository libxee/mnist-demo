FROM node:8.4
COPY . /app
WORKDIR /app
RUN RUN npm install --registry=https://registry.npm.taobao.org
EXPOSE 1234
CMD npm run start