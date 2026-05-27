import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as iam from 'aws-cdk-lib/aws-iam';

export class MachineLearningStack extends cdk.Stack {
  public readonly mlBucket: s3.Bucket;
  public readonly sagemakerRole: iam.Role;

  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    this.mlBucket = new s3.Bucket(this, 'BistroTechMlBucket', {
      bucketName: `bistrotech-ml-${this.account || 'default'}-${this.region || 'us-east-1'}`,
      versioned: true,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    this.sagemakerRole = new iam.Role(this, 'SageMakerExecutionRole', {
      roleName: `BistroTechSageMakerExecutionRole-${this.region || 'us-east-1'}`,
      assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSageMakerFullAccess'),
      ],
    });

    this.mlBucket.grantReadWrite(this.sagemakerRole);

    this.sagemakerRole.addToPolicy(new iam.PolicyStatement({
      actions: ['lambda:InvokeFunction'],
      resources: ['*'],
    }));

    // Outputs
    new cdk.CfnOutput(this, 'MlBucketName', {
      value: this.mlBucket.bucketName,
      description: 'Nombre del bucket S3 para datos y modelos de ML',
    });

    new cdk.CfnOutput(this, 'SageMakerRoleArn', {
      value: this.sagemakerRole.roleArn,
      description: 'ARN del rol de ejecucion para SageMaker',
    });
  }
}
