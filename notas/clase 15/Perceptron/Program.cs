using System;

namespace Perceptron
{ // compuerta logica AND
    class Program{
        static void Main(string[] args)
        { // {x1, x2, y}
            int[,] datos = {{1,1,1}, {1, 0, 0}, {0, 1, 0}, {0, 0, 0}};
            Random aleatorio = new Random();
            double[] pesos = {aleatorio.NextDouble(), aleatorio.NextDouble(), aleatorio.NextDouble()};
            bool aprendizaje = true;
            int salidaInt;
            int epocas = 0;
            while (aprendizaje)
            {
                aprendizaje = false;
                for (int i = 0; i < 4; i++)
                {
                    double salidaDoub = datos[i,0] * pesos[0] + datos[i,1] * pesos[1] + pesos[2];
                    if (salidaDoub > 0) salidaInt = 1; else salidaInt = 0;
                    if (salidaInt != datos[i, 2])
                    {
                        pesos[0] = aleatorio.NextDouble() - aleatorio.NextDouble();
                        pesos[1] = aleatorio.NextDouble() - aleatorio.NextDouble();
                        pesos[2] = aleatorio.NextDouble() - aleatorio.NextDouble();
                        aprendizaje = true;
                    }
                    
                }
                epocas++;
            }
        // fin del aprendizaje
        // Aqui se hacen las pruebas
        for (int i = 0; i < 4; i++)
                {
                    double salidaDoub = datos[i,0] * pesos[0] + datos[i,1] * pesos[1] + pesos[2];
                    if (salidaDoub > 0) salidaInt = 1; else salidaInt = 0;
                    Console.WriteLine("Entrada: " + datos[i,0].ToString() + " AND " + datos[i,1].ToString() + " = " + datos[i,2].ToString() + " perceptron = " + salidaInt);
                }
                Console.WriteLine("Epocas: " + epocas.ToString());
                Console.WriteLine("Pesos utiles: w0=" + pesos[0].ToString() + " w1= " + pesos[1].ToString() + " bias= " + pesos[2].ToString());
                Console.ReadLine();

        }
    }
}