package com.example.deteccionlugaresuteq.ModelTFLControl;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class TFLControl {
    float []  confidence;
    String [] etiquetas;
    ArrayList<Object[]> datosModelo;

    public float[] getConfidence() {
        return confidence;
    }

    public void setConfidence(float[] confidence) {
        this.confidence = confidence;
    }

    public String[] getEtiquetas() {
        return etiquetas;
    }

    public void setEtiquetas(String[] etiquetas) {
        this.etiquetas = etiquetas;
    }

    public ArrayList<Object[]> getDatosModelo() {
        return datosModelo;
    }

    public void setDatosModelo(ArrayList<Object[]> datosModelo) {
        this.datosModelo = datosModelo;
    }

    public String getRuta_etiquetas() {
        return ruta_etiquetas;
    }

    public void setRuta_etiquetas(String ruta_etiquetas) {
        this.ruta_etiquetas = ruta_etiquetas;
    }

    String ruta_etiquetas;
    public TFLControl(float [] confidence, String [] etiquetas) {
        this.confidence = confidence;
        this.etiquetas = etiquetas;
    }

    private void readFile() throws FileNotFoundException {
        BufferedReader reader = new BufferedReader(new FileReader(this.ruta_etiquetas));
        String line = null;
        try {

            line = reader.readLine();
            while (line != null) {
                System.out.println(line);
                line = reader.readLine();
            }
            reader.close();

        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

    public void ordenarResultados(){
        int n = confidence.length;
        for (int i = 0; i < n-1; i++) {
            for (int j = 0; j < n-i-1; j++) {
                if (confidence[j] < confidence[j+1]) {
                    float temp_probabilidad = confidence[j];
                    String temp_etiqueta = etiquetas[j];

                    confidence[j] = confidence[j + 1];
                    etiquetas[j] = etiquetas[j + 1];

                    confidence[j + 1]  = temp_probabilidad;
                    etiquetas[j + 1] = temp_etiqueta;
                }
            }
        }
    }
}
