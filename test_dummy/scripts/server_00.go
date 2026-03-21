package main

import (
	"fmt"
	"net/http"
	"encoding/json"
)

type Response struct {
	Message string `json:"message"`
	Code    int    `json:"code"`
}

func handler(w http.ResponseWriter, r *http.Request) {
	resp := Response{Message: "Hello", Code: 200}
	json.NewEncoder(w).Encode(resp)
}

func main() {
	http.HandleFunc("/", handler)
	fmt.Println("Listening on :8080")
	http.ListenAndServe(":8080", nil)
}
