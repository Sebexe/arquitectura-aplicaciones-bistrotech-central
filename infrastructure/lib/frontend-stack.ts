import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as cloudfront from 'aws-cdk-lib/aws-cloudfront';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';

export class FrontendStack extends cdk.Stack {
	constructor(scope: Construct, id: string, props?: cdk.StackProps) {
		super(scope, id, props);

		const siteBucket = new s3.Bucket(this, 'BellavistaRestauranteBucket', {
			bucketName: 'bellavista-restaurante',
			blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
			enforceSSL: true,
			encryption: s3.BucketEncryption.S3_MANAGED,
			versioned: false,
			publicReadAccess: false,
		});

		const originAccessControl = new cloudfront.CfnOriginAccessControl(this, 'BellavistaRestauranteOAC', {
			originAccessControlConfig: {
				name: 'bellavista-restaurante-oac',
				description: 'OAC for bellavista-restaurante S3 origin',
				originAccessControlOriginType: 's3',
				signingBehavior: 'always',
				signingProtocol: 'sigv4',
			},
		});

		const distribution = new cloudfront.CfnDistribution(this, 'BellavistaRestauranteDistribution', {
			distributionConfig: {
				enabled: true,
				defaultRootObject: 'index.html',
				origins: [
					{
						id: 'BellavistaRestauranteS3Origin',
						domainName: siteBucket.bucketRegionalDomainName,
						originAccessControlId: originAccessControl.attrId,
						s3OriginConfig: {
							originAccessIdentity: '',
						},
					},
				],
				defaultCacheBehavior: {
					targetOriginId: 'BellavistaRestauranteS3Origin',
					viewerProtocolPolicy: 'redirect-to-https',
					allowedMethods: ['GET', 'HEAD', 'OPTIONS'],
					cachedMethods: ['GET', 'HEAD'],
					compress: true,
					cachePolicyId: cloudfront.CachePolicy.CACHING_OPTIMIZED.cachePolicyId,
					  originRequestPolicyId: cloudfront.OriginRequestPolicy.CORS_S3_ORIGIN.originRequestPolicyId,
				},
				priceClass: 'PriceClass_100',
				restrictions: {
					geoRestriction: {
						restrictionType: 'none',
						locations: [],
					},
				},
				viewerCertificate: {
					cloudFrontDefaultCertificate: true,
				},
			},
		});

		siteBucket.addToResourcePolicy(
			new iam.PolicyStatement({
				actions: ['s3:GetObject'],
				resources: [siteBucket.arnForObjects('*')],
				principals: [new iam.ServicePrincipal('cloudfront.amazonaws.com')],
				conditions: {
					StringEquals: {
						'AWS:SourceArn': `arn:aws:cloudfront::${cdk.Aws.ACCOUNT_ID}:distribution/${distribution.ref}`,
					},
				},
			}),
		);

		new cdk.CfnOutput(this, 'FrontendBucketName', {
			value: siteBucket.bucketName,
		});

		new cdk.CfnOutput(this, 'FrontendDistributionDomainName', {
			value: distribution.attrDomainName,
		});

		new cdk.CfnOutput(this, 'FrontendDistributionId', {
			value: distribution.ref,
		});
	}
}
