import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable } from 'rxjs';
import { environment } from './../environments/environment';


export interface Prediction {
  prediction: string;
}

@Injectable({
  providedIn: 'root'
})
export class PredictionService {

  constructor(private http: HttpClient) { }

  public getPrediction(input: string): Observable<Prediction> {
    return this.http.post<Prediction>(`${environment.apiUrl}/predict`, { input: input })
  }
}
