#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { BackendStack } from '../lib/backend-stack';
import { FrontendStack } from '../lib/frontend-stack';
import { MachineLearningStack } from '../lib/machine-learning-stack';

declare const process: {
	env: {
		CDK_DEFAULT_ACCOUNT?: string;
		CDK_DEFAULT_REGION?: string;
	};
};

const app = new cdk.App();
const env = {
	account: process.env.CDK_DEFAULT_ACCOUNT,
	region: process.env.CDK_DEFAULT_REGION,
};

new BackendStack(app, 'BackendStack', { env });

new FrontendStack(app, 'FrontendStack', { env });

new MachineLearningStack(app, 'MachineLearningStack', { env });