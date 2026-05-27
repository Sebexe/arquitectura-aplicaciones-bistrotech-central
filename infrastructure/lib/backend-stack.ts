import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as path from 'path';
// import * as sqs from 'aws-cdk-lib/aws-sqs';

export class BackendStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);
    const backendAssetDir = this.node.tryGetContext('backendAssetDir') ?? path.join(__dirname, '../../app/backend');

    new dynamodb.Table(this, 'MesasTable', {
      partitionKey: { name: 'id_mesa', type: dynamodb.AttributeType.NUMBER },
      tableName: 'bistrotech-mesas',
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
    });

    new dynamodb.Table(this, 'ReservasTable', {
      partitionKey: { name: 'id_reserva', type: dynamodb.AttributeType.STRING },
      tableName: 'bistrotech-reservas',
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
    });

    new dynamodb.Table(this, 'RegistrosTable', {
      partitionKey: { name: 'id_mesa', type: dynamodb.AttributeType.NUMBER },
      sortKey: { name: 'persona_ts', type: dynamodb.AttributeType.STRING },
      tableName: 'bistrotech-registros',
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
    });

    new dynamodb.Table(this, 'ClientesHistoricoTable', {
      partitionKey: { name: 'id_cliente', type: dynamodb.AttributeType.NUMBER },
      tableName: 'bistrotech-clientes-historico',
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
    });

    new dynamodb.Table(this, 'SegmentosReferenciaTable', {
      partitionKey: { name: 'segmento_pk', type: dynamodb.AttributeType.STRING },
      tableName: 'bistrotech-segmentos-referencia',
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
    });

    const adapterLayer = lambda.LayerVersion.fromLayerVersionArn(
      this,
      'LambdaAdapterLayer',
      'arn:aws:lambda:us-east-1:753240598075:layer:LambdaAdapterLayerX86:27',
    );

    new lambda.Function(this, 'BackendLambda', {
      functionName: "bistrotech-backend-lambda-function",
      runtime: lambda.Runtime.PYTHON_3_13,
      architecture: lambda.Architecture.X86_64,
      handler: 'run.sh',
      code: lambda.Code.fromAsset(backendAssetDir),
      timeout: cdk.Duration.seconds(10),
      memorySize: 512,
      layers: [adapterLayer],
      environment: {
        AWS_LAMBDA_EXEC_WRAPPER: '/opt/bootstrap',
        AWS_LWA_PORT: '8000',
      },
    });

    
  }
}

// TABLES = [
//     {
//         "TableName": "bistrotech-mesas",
//         "KeySchema": [
//             {"AttributeName": "id_mesa", "KeyType": "HASH"},
//         ],
//         "AttributeDefinitions": [
//             {"AttributeName": "id_mesa", "AttributeType": "N"},
//         ],
//         "BillingMode": "PAY_PER_REQUEST",
//         "comment": "Una fila por mesa física. Estado: libre/ocupada/reservada. Soft delete con activa=False.",
//     },
//     {
//         "TableName": "bistrotech-reservas",
//         "KeySchema": [
//             {"AttributeName": "id_reserva", "KeyType": "HASH"},
//         ],
//         "AttributeDefinitions": [
//             {"AttributeName": "id_reserva", "AttributeType": "S"},
//         ],
//         "BillingMode": "PAY_PER_REQUEST",
//         "comment": "Una fila por reserva. id_reserva es UUID. Estado: confirmada/cancelada/completada.",
//     },
//     {
//         "TableName": "bistrotech-registros",
//         "KeySchema": [
//             {"AttributeName": "id_mesa",    "KeyType": "HASH"},
//             {"AttributeName": "persona_ts", "KeyType": "RANGE"},
//         ],
//         "AttributeDefinitions": [
//             {"AttributeName": "id_mesa",    "AttributeType": "N"},
//             {"AttributeName": "persona_ts", "AttributeType": "S"},
//         ],
//         "BillingMode": "PAY_PER_REQUEST",
//         "comment": (
//             "Una fila por comensal por visita. "
//             "persona_ts = '{id_persona_en_mesa}#{ISO-timestamp}'. "
//             "Sin columnas de feedback — esas se actualizan desde el POS post-servicio."
//         ),
//     },
//     {
//         "TableName": "bistrotech-clientes-historico",
//         "KeySchema": [
//             {"AttributeName": "id_cliente", "KeyType": "HASH"},
//         ],
//         "AttributeDefinitions": [
//             {"AttributeName": "id_cliente", "AttributeType": "N"},
//         ],
//         "BillingMode": "PAY_PER_REQUEST",
//         "comment": (
//             "Perfil acumulado por cliente identificado. "
//             "Campos: visitas_totales, ticket_promedio, restriccion_detectada, "
//             "motivo_frecuente, franja_horaria_frecuente, like_rate_promedio, platos_frecuentes."
//         ),
//     },
//     {
//         "TableName": "bistrotech-segmentos-referencia",
//         "KeySchema": [
//             {"AttributeName": "segmento_pk", "KeyType": "HASH"},
//         ],
//         "AttributeDefinitions": [
//             {"AttributeName": "segmento_pk", "AttributeType": "S"},
//         ],
//         "BillingMode": "PAY_PER_REQUEST",
//         "comment": (
//             "Medias por segmento para cold start. "
//             "segmento_pk = '{franja_etaria}#{franja_horaria}#{motivo_visita}'. "
//             "Campos: ticket_promedio_segmento, platos_populares_segmento, propina_rate_segmento."
//         ),
//     },
// ]



