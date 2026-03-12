package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"github.com/gin-gonic/gin"
)

type Query struct {
	Question        string  `json:"question"`
	TopK            int     `json:"top_k,omitempty"`
	LanguageFilter  *string `json:"language_filter,omitempty"`
}

type Response struct {
	Answer  string      `json:"answer"`
	Sources []struct {
		ID    string  `json:"id"`
		Name  string  `json:"name"`
		Score float32 `json:"score"`
	} `json:"sources"`
}

func main() {
	r := gin.Default()
	r.POST("/api/ask", func(c *gin.Context) {
		var q Query
		if err := c.BindJSON(&q); err != nil {
			c.JSON(400, gin.H{"error": err.Error()})
			return
		}

		if q.TopK == 0 { q.TopK = 8 }

		payload := map[string]any{
			"question": q.Question,
			"top_k":    q.TopK,
		}
		if q.LanguageFilter != nil {
			payload["language_filter"] = *q.LanguageFilter
		}

		body, _ := json.Marshal(payload)
		resp, err := http.Post("http://127.0.0.1:8001/ask", "application/json", bytes.NewBuffer(body))
		if err != nil {
			// fmt.Println(err)
			c.JSON(500, gin.H{"error": "RAG service down"})
			return
		}
		defer resp.Body.Close()

		data, _ := io.ReadAll(resp.Body)
		var result Response
		json.Unmarshal(data, &result)

		c.JSON(200, result)
	})

	r.Run(":8888")
}