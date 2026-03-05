package com.example.photogenerate

import android.content.Context
import org.json.JSONObject
import android.util.Log

class Tokenizer(context: Context) {
    private val vocab = mutableMapOf<String, Int>()
    private val maxTokenLength = 77

    // Специальные ID для Stable Diffusion 1.5
    private val startToken = 49406L
    private val endToken = 49407L

    init {
        try {
            // 1. Читаем файл как текст из assets
            val jsonString = context.assets.open("tokenizer.json").bufferedReader().use { it.readText() }
            val jsonObject = JSONObject(jsonString)

            // 2. В стандартном формате HuggingFace словарь лежит в "model" -> "vocab"
            val modelObj = jsonObject.getJSONObject("model")
            val vocabJson = modelObj.getJSONObject("vocab")

            // 3. Переносим ключи в Map для быстрого поиска
            val keys = vocabJson.keys()
            while (keys.hasNext()) {
                val key = keys.next()
                vocab[key] = vocabJson.getInt(key)
            }
            Log.d("SD_DEBUG", "Словарь загружен: ${vocab.size} слов")
        } catch (e: Exception) {
            Log.e("SD_ERROR", "Ошибка загрузки tokenizer.json: ${e.message}")
        }
    }

    fun tokenize(text: String): LongArray {
        // Простая очистка текста и разбиение на слова
        val words = text.lowercase()
            .replace(Regex("[^a-z0-9\\s]"), "") // Убираем знаки препинания
            .split(Regex("\\s+"))
            .filter { it.isNotEmpty() }

        val result = LongArray(maxTokenLength) { endToken } // Заполняем padding (49407)

        result[0] = startToken // Начальный токен (49406)

        var currentIndex = 1
        for (word in words) {
            if (currentIndex >= maxTokenLength - 1) break

            // Ищем ID слова. Если слова нет — используем 49407 или можно искать подстроки (BPE)
            // Для базовой работы этого достаточно
            val id = vocab[word] ?: vocab["<|endoftext|>"] ?: 49407
            result[currentIndex] = id.toLong()
            currentIndex++
        }

        // В конце всегда ставим токен завершения, если есть место
        if (currentIndex < maxTokenLength) {
            result[currentIndex] = endToken
        }

        return result
    }
}