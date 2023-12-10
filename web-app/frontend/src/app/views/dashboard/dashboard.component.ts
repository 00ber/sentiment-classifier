import { Component } from '@angular/core';
import { PredictionService } from "../../predictions.service"
import { Subject } from 'rxjs';

export interface TableRow {
  text: string;
  prediction: string;
}

@Component({
  templateUrl: 'dashboard.component.html',
  styleUrls: ['dashboard.component.scss']
})
export class DashboardComponent{

  constructor(private predictionsApi: PredictionService) {}

  public userInput: string = "";
  public prediction: string = "";
  public displayPrediction = false;
  public displayClass = ""

  public tableRows: TableRow[] = [];
  public tableRows$ = new Subject<TableRow[]>();


  public clearInputs(): void {
    this.userInput = "";
    this.displayPrediction = false;
  }

  public analyze(): void {
    if (!this.userInput) {
      return;
    }
    this.predictionsApi.getPrediction(this.userInput).subscribe(data => {
      this.prediction = data.prediction;
      this.displayClass = data.prediction === "positive" ? "success" : "danger";
      this.displayPrediction = true;
  
      this.tableRows.unshift({
          text: this.userInput,
          prediction: data.prediction
      });
      this.tableRows = this.tableRows.slice(0, 10);
      this.tableRows$.next(this.tableRows);
    })
  }
}
