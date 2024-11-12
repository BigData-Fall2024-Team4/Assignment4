FROM node:20-slim AS base
ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"
RUN corepack enable
COPY . /app
WORKDIR /app

# Builder Stage
FROM base AS builder
WORKDIR /app
COPY package*.json ./
RUN pnpm install --frozen-lockfile
COPY . .
COPY .env ./
RUN pnpm run build

# Production Stage 
FROM base AS production
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter
WORKDIR /app
# Copy the built artifacts from the builder stage
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY .env ./
ENV NODE_ENV=production
ENV REMOTE_ACTION_URL=http://agent:8000/copilotkit
EXPOSE 3000
CMD ["node", "server.js"]