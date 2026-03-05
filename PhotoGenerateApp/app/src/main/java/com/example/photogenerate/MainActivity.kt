package com.example.photogenerate

import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    // Объявляем наш движок
    private lateinit var sdRunner: SDRunner

    // Элементы интерфейса
    private lateinit var promptInput: EditText
    private lateinit var generateButton: Button
    private lateinit var resultImage: ImageView
    private lateinit var progressBar: ProgressBar

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Привязываем UI
        promptInput = findViewById(R.id.promptInput)
        generateButton = findViewById(R.id.generateButton)
        resultImage = findViewById(R.id.resultImage)
        progressBar = findViewById(R.id.progressBar)

        // Блокируем кнопку, пока модели грузятся
        generateButton.isEnabled = false

        // 1. Инициализируем бэкенд в фоновом потоке
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // Создаем экземпляр нашего "тяжелого" класса
                sdRunner = SDRunner(this@MainActivity)

                withContext(Dispatchers.Main) {
                    generateButton.isEnabled = true
                    Toast.makeText(this@MainActivity, "Модели загружены в NPU!", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                android.util.Log.e("SD_ERROR", "!!! ОШИБКА ЗАГРУЗКИ МОДЕЛЕЙ !!!")
                android.util.Log.e("SD_ERROR", "Причина: ${e.message}")
                android.util.Log.e("SD_ERROR", "Стэк: ${e.stackTraceToString()}")
                e.printStackTrace() // Это выведет подробный путь ошибки (Stacktrace) ь67
                withContext(Dispatchers.Main) {
                    // Создаем большое окно с текстом ошибки
                    android.app.AlertDialog.Builder(this@MainActivity)
                        .setTitle("Ой! Что-то пошло не так")
                        .setMessage(e.stackTraceToString()) // Это вывалит всю техническую инфу
                        .setPositiveButton("Понятно", null)
                        .show()
                }
            }
        }

        // 2. Обработка нажатия на кнопку
        generateButton.setOnClickListener {
            val prompt = promptInput.text.toString()
            if (prompt.isNotBlank()) {
                startGeneration(prompt)
            }
        }
    }

    private fun startGeneration(prompt: String) {
        // Показываем прогресс и блокируем кнопку
        progressBar.visibility = View.VISIBLE
        progressBar.progress = 0
        generateButton.isEnabled = false

        // Запускаем генерацию (в Dispatchers.IO, чтобы не вешать экран)
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // Вызываем наш SDRunner
                val bitmap = sdRunner.generate(prompt) { progress ->
                    // Обновляем прогресс-бар в реальном времени
                    runOnUiThread { progressBar.progress = progress }
                }

                // Выводим результат на экран
                withContext(Dispatchers.Main) {
                    resultImage.setImageBitmap(bitmap)
                    finalizeUi()
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    android.util.Log.e("SD_ERROR", "!!! ОШИБКА ГЕНЕРАЦИИ !!!")
                    android.util.Log.e("SD_ERROR", "Причина: ${e.message}")
                    android.util.Log.e("SD_ERROR", "Стэк: ${e.stackTraceToString()}")
                    Toast.makeText(this@MainActivity, "Ошибка генерации: ${e.message}", Toast.LENGTH_LONG).show()
                    finalizeUi()
                }
            }
        }
    }

    private fun finalizeUi() {
        progressBar.visibility = View.GONE
        generateButton.isEnabled = true
    }
}