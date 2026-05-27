import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as path from 'path';
// import * as sqs from 'aws-cdk-lib/aws-sqs';

export class BackendStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);
    const backendAssetDir = this.node.tryGetContext('backendAssetDir') ?? path.join(__dirname, '../../app/backend');

    const mesasTable = new dynamodb.Table(this, 'MesasTable', {
      partitionKey: { name: 'id_mesa', type: dynamodb.AttributeType.NUMBER },
      tableName: 'bistrotech-mesas',
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
    });

    const reservasTable = new dynamodb.Table(this, 'ReservasTable', {
      partitionKey: { name: 'id_reserva', type: dynamodb.AttributeType.STRING },
      tableName: 'bistrotech-reservas',
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
    });

    const registrosTable = new dynamodb.Table(this, 'RegistrosTable', {
      partitionKey: { name: 'id_mesa', type: dynamodb.AttributeType.NUMBER },
      sortKey: { name: 'persona_ts', type: dynamodb.AttributeType.STRING },
      tableName: 'bistrotech-registros',
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
    });

    const clientesHistoricoTable = new dynamodb.Table(this, 'ClientesHistoricoTable', {
      partitionKey: { name: 'id_cliente', type: dynamodb.AttributeType.NUMBER },
      tableName: 'bistrotech-clientes-historico',
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
    });

    const segmentosReferenciaTable = new dynamodb.Table(this, 'SegmentosReferenciaTable', {
      partitionKey: { name: 'segmento_pk', type: dynamodb.AttributeType.STRING },
      tableName: 'bistrotech-segmentos-referencia',
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
    });

    const backendRole = new iam.Role(this, 'BackendLambdaRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole'),
      ],
    });

    const adapterLayer = lambda.LayerVersion.fromLayerVersionArn(
      this,
      'LambdaAdapterLayer',
      'arn:aws:lambda:us-east-1:753240598075:layer:LambdaAdapterLayerX86:27',
    );

    const backendLambda = new lambda.Function(this, 'BackendLambda', {
      functionName: "bistrotech-backend-lambda-function",
      runtime: lambda.Runtime.PYTHON_3_13,
      architecture: lambda.Architecture.X86_64,
      handler: 'run.sh',
      code: lambda.Code.fromAsset(backendAssetDir),
      role: backendRole,
      timeout: cdk.Duration.seconds(10),
      memorySize: 512,
      layers: [adapterLayer],
      environment: {
        AWS_LAMBDA_EXEC_WRAPPER: '/opt/bootstrap',
        AWS_LWA_PORT: '8000',
      },
    });

    const backendFunctionUrl = backendLambda.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.NONE,
      cors: {
        allowedOrigins: ['*'],
        allowedMethods: [lambda.HttpMethod.ALL],
        allowedHeaders: ['*'],
      },
    });

    new cdk.CfnOutput(this, 'BackendFunctionUrl', {
      value: backendFunctionUrl.url,
    });

    for (const table of [mesasTable, reservasTable, registrosTable, clientesHistoricoTable, segmentosReferenciaTable]) {
      table.grantReadWriteData(backendLambda);
    }

    
  }
}

