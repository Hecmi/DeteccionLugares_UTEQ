package com.example.deteccionlugaresuteq;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraManager;
import android.media.Image;
import android.media.ImageReader;
import android.os.Build;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import com.example.deteccionlugaresuteq.ModelTFLControl.TFLControl;
import com.example.deteccionlugaresuteq.ml.ModelLugaresUteq0;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.mlkit.vision.text.Text;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import camerax.CameraConnectionFragment;
import camerax.ImageUtils;

public class MainActivity extends AppCompatActivity
        implements OnSuccessListener<Text>,
        OnFailureListener, ImageReader.OnImageAvailableListener {
    public static int REQUEST_CAMERA = 111;
    public static int REQUEST_GALLERY = 222;
    private float PROBABILIDAD_VALIDA = 95;
    private int TAMANIO_IMAGEN = 224;
    ArrayList<String> permisosNoAprobados;
    Button btnCamara;
    Button btnGaleria;
    TextView txtResults, txtPorcentajeValido;
    TextToSpeech tts;

    String prediccionModelo;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ArrayList<String> permisos_requeridos = new ArrayList<String>();
        permisos_requeridos.add(android.Manifest.permission.CAMERA);
        permisos_requeridos.add(android.Manifest.permission.MANAGE_EXTERNAL_STORAGE);
        permisos_requeridos.add(android.Manifest.permission.READ_EXTERNAL_STORAGE);

        txtResults = findViewById(R.id.txtresults);
        txtPorcentajeValido = findViewById(R.id.txtPorcentajeValido);

        btnCamara = findViewById(R.id.btCamera);
        //btnGaleria=  findViewById(R.id.btGallery);
        permisosNoAprobados  = getPermisosNoAprobados(permisos_requeridos);
        requestPermissions(permisosNoAprobados.toArray(new String[permisosNoAprobados.size()]),
                100);


        //Instanciar la variable para la gestión del texto a voz.
        tts = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if (status != TextToSpeech.ERROR) tts.setLanguage(new Locale("SPA", "MEX"));
            }
        });
    }

    public void Reconocer(Bitmap imagen){
        try {
            //Instanciar el modelo basado en el creado (tensor flow lite)
            ModelLugaresUteq0 model = ModelLugaresUteq0.newInstance(getApplicationContext());

            //Definir un bitmap para enviarlo al modelo creado.
            imagen = Bitmap.createScaledBitmap(imagen, TAMANIO_IMAGEN, TAMANIO_IMAGEN, true);

            //Establecer las dimensiones que tendrá la imágen.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, TAMANIO_IMAGEN, TAMANIO_IMAGEN, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * TAMANIO_IMAGEN * TAMANIO_IMAGEN * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            //Obtener los pixeles de la imagen e iterar cada uno de los pixeles para obtener el color respectivo
            //en formato RGB (Red, green, blue).
            int[] intValues = new int[TAMANIO_IMAGEN * TAMANIO_IMAGEN];
            imagen.getPixels(intValues, 0, imagen.getWidth(), 0, 0, imagen.getWidth(), imagen.getHeight());

            int pixel = 0;

            for(int i = 0; i <  imagen.getHeight(); i ++){
                for(int j = 0; j < imagen.getWidth(); j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }
            }

            //Cargar los bytes obtenidos al modelo de tensorflow.
            inputFeature0.loadBuffer(byteBuffer);

            ModelLugaresUteq0.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            //Obtener los resultados correspondientes a las neuronas existentes en el modelo.
            float[] confidences = outputFeature0.getFloatArray();
            //String[] classes = {"Gerard Way", "Brendon Urie", "Freddie Mercury"};
            String[] classes = {"Polideportivo", "FCE", "Rotonda"};

            TFLControl controlModeloTFl = new TFLControl(confidences, classes);
            controlModeloTFl.ordenarResultados();

            //Obtener el que tiene la probabilidad más alta
            String [] etiquetasOrdenadas = controlModeloTFl.getEtiquetas();
            float [] probabilidadesOrdenadas = controlModeloTFl.getConfidence();

            String textoActual = txtResults.getText().toString();
            String resultado = "";

            if (etiquetasOrdenadas.length > 0){
                if (probabilidadesOrdenadas[0] * 100 > PROBABILIDAD_VALIDA){
                    resultado = etiquetasOrdenadas[0];
                    prediccionModelo = resultado;
                    if (textoActual != resultado) tts.speak(resultado, TextToSpeech.QUEUE_FLUSH, null);
                }
            }

            txtResults.setText(resultado);

            //Finalmente, cerrar el modelo.
            model.close();
        } catch (Exception e) {
            txtResults.setText(e.getMessage());
        }
    }

    public void abrirGaleria (View view){
        Intent i = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(i, REQUEST_GALLERY);
    }
    public void abrirCamera (View view){
        //Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        //startActivityForResult(intent, REQUEST_CAMERA);
        PROBABILIDAD_VALIDA = Float.parseFloat(txtPorcentajeValido.getText().toString());
        Log.i("TEST_0", "PRUEBA_");
        this.setFragment();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        for(int i=0; i<permissions.length; i++){
            if(permissions[i].equals(android.Manifest.permission.CAMERA)){
                btnCamara.setEnabled(grantResults[i] == PackageManager.PERMISSION_GRANTED);
            } else if(permissions[i].equals(android.Manifest.permission.MANAGE_EXTERNAL_STORAGE) ||
                    permissions[i].equals(android.Manifest.permission.READ_EXTERNAL_STORAGE)
            ) {
                //btnGaleria.setEnabled(grantResults[i] == PackageManager.PERMISSION_GRANTED);
            }
        }
    }

    public ArrayList<String> getPermisosNoAprobados(ArrayList<String>  listaPermisos) {
        ArrayList<String> list = new ArrayList<String>();
        Boolean habilitado;


        if (Build.VERSION.SDK_INT >= 23)
            for(String permiso: listaPermisos) {
                if (checkSelfPermission(permiso) != PackageManager.PERMISSION_GRANTED) {
                    list.add(permiso);
                    habilitado = false;
                }else
                    habilitado=true;

                if(permiso.equals(android.Manifest.permission.CAMERA))
                    btnCamara.setEnabled(habilitado);
                else if (permiso.equals(android.Manifest.permission.MANAGE_EXTERNAL_STORAGE)  ||
                        permiso.equals(Manifest.permission.READ_EXTERNAL_STORAGE)) continue;
                //btnGaleria.setEnabled(habilitado);
            }


        return list;
    }

    @Override
    public void onFailure(@NonNull Exception e) {
        txtResults.setText("Error al procesar imagen");
    }

    @Override
    public void onSuccess(Text text) {
        List<Text.TextBlock> blocks = text.getTextBlocks();
        String resultados="";
        if (blocks.size() == 0) {
            resultados = "No hay Texto";
        }else{
            for (int i = 0; i < blocks.size(); i++) {
                List<Text.Line> lines = blocks.get(i).getLines();
                for (int j = 0; j < lines.size(); j++) {
                    List<Text.Element> elements = lines.get(j).getElements();
                    for (int k = 0; k < elements.size(); k++) {
                        resultados = resultados + elements.get(k).getText() + " ";
                    }
                }
            }
            resultados=resultados + "\n";
        }
        txtResults.setText(resultados);
    }

    int previewHeight = 0, previewWidth = 0;
    int sensorOrientation;

    protected void setFragment() {
        Log.i("TEST_0", "PRUEBA_");
        final CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        String cameraId = null;
        try {
            cameraId = manager.getCameraIdList()[0];
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
        Log.i("TEST_1", "PRUEBA_");
        CameraConnectionFragment fragment;
        CameraConnectionFragment camera2Fragment =
                CameraConnectionFragment.newInstance(
                        new CameraConnectionFragment.ConnectionCallback() {
                            @Override
                            public void onPreviewSizeChosen(final Size size, final int rotation) {
                                previewHeight = size.getHeight();    previewWidth = size.getWidth();
                                sensorOrientation = rotation - getScreenOrientation();
                            }
                        }, this,   R.layout.camerafragment, new Size(640, 480));

        camera2Fragment.setCamera(cameraId);
        fragment = camera2Fragment;
        getFragmentManager().beginTransaction().replace(R.id.container, fragment).commit();
        Log.i("TEST_2", "PRUEBA_");
    }
    protected int getScreenOrientation() {
        switch (getWindowManager().getDefaultDisplay().getRotation()) {
            case Surface.ROTATION_270:
                return 270;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_90:
                return 90;
            default:
                return 0;
        }
    }
    private boolean isProcessingFrame = false;
    private byte[][] yuvBytes = new byte[3][];
    private int[] rgbBytes = null;
    private int yRowStride;
    private Runnable postInferenceCallback;
    private Runnable imageConverter;
    private Bitmap rgbFrameBitmap;
    @Override
    public void onImageAvailable(ImageReader imageReader) {
        if (previewWidth == 0 || previewHeight == 0)           return;
        if (rgbBytes == null)    rgbBytes = new int[previewWidth * previewHeight];
        try {
            final Image image = imageReader.acquireLatestImage();
            if (image == null)    return;
            if (isProcessingFrame) {           image.close();            return;         }
            isProcessingFrame = true;
            final Image.Plane[] planes = image.getPlanes();
            fillBytes(planes, yuvBytes);
            yRowStride = planes[0].getRowStride();
            final int uvRowStride = planes[1].getRowStride();
            final int uvPixelStride = planes[1].getPixelStride();
            imageConverter =  new Runnable() {
                @Override
                public void run() {
                    ImageUtils.convertYUV420ToARGB8888( yuvBytes[0], yuvBytes[1], yuvBytes[2], previewWidth,  previewHeight,
                            yRowStride,uvRowStride, uvPixelStride,rgbBytes);
                }
            };
            postInferenceCallback =      new Runnable() {
                @Override
                public void run() {  image.close(); isProcessingFrame = false;  }
            };

            processImage();

        } catch (final Exception e) {    }

    }
    protected void fillBytes(final Image.Plane[] planes, final byte[][] yuvBytes) {
        for (int i = 0; i < planes.length; ++i) {
            final ByteBuffer buffer = planes[i].getBuffer();
            if (yuvBytes[i] == null) {
                yuvBytes[i] = new byte[buffer.capacity()];
            }
            buffer.get(yuvBytes[i]);
        }
    }
    private void processImage() {
        imageConverter.run();

        //Adaptar la imagen al formato requerido por el modelo
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
        rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);

        try {
            Reconocer(rgbFrameBitmap);
            /*Modelomascota model = Modelomascota.newInstance(getApplicationContext());
            TensorImage image = TensorImage.fromBitmap(rgbFrameBitmap);

            Modelomascota.Outputs outputs = model.process(image);
            List<Category> probability = outputs.getProbabilityAsCategoryList();
            Collections.sort(probability, new CategoryComparator());

            //Una vez es ordenado el vector de categorías la posición 0 es la que contiene la mayor probabilidad.
            String prediccion = "";
            Float probabilidad = 0f;

            String textoActual = txtResults.getText().toString();
            prediccion = probability.get(0).getLabel();
            probabilidad = probability.get(0).getScore();

            //Verificar que la probabilidad más alta sea mayor a un valor específico.
            if (probabilidad * 100 > PROBABILIDAD_VALIDA)
                if (textoActual != prediccion) {
                    prediccionModelo = prediccion;
                    txtResults.setText(prediccion);
                    tts.speak(prediccion, TextToSpeech.QUEUE_FLUSH, null);
                }


            model.close();*/
        } catch (Exception e) {
            txtResults.setText("Error al procesar Modelo");
        }
        postInferenceCallback.run();
    }

    public void btn_LeerTexto(View view){
        tts.speak(prediccionModelo, TextToSpeech.QUEUE_FLUSH, null, null);
    }
}

