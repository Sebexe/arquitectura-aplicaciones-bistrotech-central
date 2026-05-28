import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as acm from 'aws-cdk-lib/aws-certificatemanager';
import * as apigatewayv2 from 'aws-cdk-lib/aws-apigatewayv2';
import * as integrations from 'aws-cdk-lib/aws-apigatewayv2-integrations';
import * as route53 from 'aws-cdk-lib/aws-route53';
import * as targets from 'aws-cdk-lib/aws-route53-targets';
import * as path from 'path';
// import * as sqs from 'aws-cdk-lib/aws-sqs';

export class BackendStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);
    const backendAssetDir = this.node.tryGetContext('backendAssetDir') ?? path.join(__dirname, '../../app/backend');
    const hostedZoneDomainName = this.node.tryGetContext('hostedZoneDomainName') ?? 'bellavista.sebastianzafra.com';
    const apiDomainName = this.node.tryGetContext('apiDomainName') ?? `api.${hostedZoneDomainName}`;
    const apiRecordName = apiDomainName.endsWith(`.${hostedZoneDomainName}`)
      ? apiDomainName.slice(0, -(hostedZoneDomainName.length + 1))
      : apiDomainName;

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

    const hostedZone = route53.HostedZone.fromLookup(this, 'BackendHostedZone', {
      domainName: hostedZoneDomainName,
    });

    const certificate = new acm.Certificate(this, 'BackendApiCertificate', {
      domainName: apiDomainName,
      validation: acm.CertificateValidation.fromDns(hostedZone),
    });

    const apiDomain = new apigatewayv2.DomainName(this, 'BackendApiDomain', {
      domainName: apiDomainName,
      certificate,
    });

    const backendApi = new apigatewayv2.HttpApi(this, 'BackendHttpApi', {
      apiName: 'bistrotech-backend-api',
      defaultDomainMapping: {
        domainName: apiDomain,
      },
    });

    const backendIntegration = new integrations.HttpLambdaIntegration(
      'BackendLambdaIntegration',
      backendLambda,
    );

    backendApi.addRoutes({
      path: '/',
      methods: [apigatewayv2.HttpMethod.ANY],
      integration: backendIntegration,
    });

    backendApi.addRoutes({
      path: '/{proxy+}',
      methods: [apigatewayv2.HttpMethod.ANY],
      integration: backendIntegration,
    });

    new route53.ARecord(this, 'BackendApiAliasRecord', {
      zone: hostedZone,
      recordName: apiRecordName,
      target: route53.RecordTarget.fromAlias(
        new targets.ApiGatewayv2DomainProperties(
          apiDomain.regionalDomainName,
          apiDomain.regionalHostedZoneId,
        ),
      ),
    });

    new cdk.CfnOutput(this, 'BackendApiUrl', {
      value: backendApi.url ?? '',
    });

    new cdk.CfnOutput(this, 'BackendApiCustomDomainUrl', {
      value: `https://${apiDomainName}`,
    });

    for (const table of [mesasTable, reservasTable, registrosTable, clientesHistoricoTable, segmentosReferenciaTable]) {
      table.grantReadWriteData(backendLambda);
    }

    
  }
}

